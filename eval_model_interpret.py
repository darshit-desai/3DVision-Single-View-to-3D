import argparse
import time
import torch
from torchvision import transforms
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from utils import get_mesh_renderer, get_points_renderer
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

import imageio

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--vis_freq', default=1000, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def evaluate(predictions, mesh_gt, thresholds, args):
    if args.type == "vox":
        voxels_src = predictions
        # voxels_src = torch.nn.Sigmoid(voxels_src)
        # predictions = predictions.squeeze(0)
        print("Prediction information:")
        print("Number of vertices:", len(predictions))
        print("Number of faces:", len(predictions))
        H,W,D = voxels_src.shape[2:]
        print("Shape of voxels_src:", voxels_src.shape)
        # vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.5)
        vertices_src, faces_src = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), isovalue=0.3)
        # USE CUBIFY
        # cubes = pytorch3d.ops.cubify(voxels_src.squeeze(0), thresh=0.0)
        # vertices_src = cubes.verts_packed()
        # faces_src = cubes.faces_packed()
        #####
        vertices_src = torch.tensor(vertices_src).float()
        print("Length of vertices_src:", len(vertices_src))
        print("Length of faces_src:", len(faces_src))
        ###############
        # USE CUBIFY
        # faces_src = torch.tensor(faces_src.detach().cpu().numpy().astype(int))
        # mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src.to(args.device)])
        ############
        faces_src = torch.tensor(faces_src.astype(int))
        mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])

        pred_points = sample_points_from_meshes(mesh_src, args.n_points)
        pred_points = utils_vox.Mem2Ref(pred_points.detach().cpu(), H, W, D)
    elif args.type == "point":
        pred_points = predictions.cpu()
    elif args.type == "mesh":
        pred_points = sample_points_from_meshes(predictions, args.n_points).cpu()

    gt_points = sample_points_from_meshes(mesh_gt, args.n_points)
    
    metrics = compute_sampling_metrics(pred_points, gt_points, thresholds)
    return metrics

def render_voxels(optimized_voxel, output_path):
    voxels_src = optimized_voxel
    # voxels_src = torch.nn.Sigmoid(voxels_src)
    voxel_size = 32
    max_value = 1.1
    min_value = -1.1
    #make vertices and faces for symmetric 360 degree rotation
    print(voxels_src.shape)
    vertices, faces = mcubes.marching_cubes(voxels_src.detach().cpu().squeeze().numpy(), 0.5)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    color1 = [0.7, 0.0, 0.4]
    color2 = [0.6, 1.0, 1.0]
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)
    print("Shape of vertices:", vertices.shape)
    z_min = vertices[:,:,2].min()
    z_max = vertices[:,:,2].max()
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    new_colors = alpha[:, :, None] * torch.tensor(color2) + (1 - alpha[:, :, None]) * torch.tensor(color1)
    textures = pytorch3d.renderer.TexturesVertex(new_colors)
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=args.device)
    voxel_chair_mesh = pytorch3d.structures.Meshes(verts=vertices, faces=faces, textures=textures).to(
        args.device
    )
    renderer = get_mesh_renderer(image_size=512, device=args.device)
    num_frames = 36
    render_full = []
    camera_positions = []
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 3.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(voxel_chair_mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=12.0, loop=0)

def render_points(optimized_points, output_path, type_data):
    image_size= 512
    background_color=(1, 1, 1)
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts = optimized_points
    rgb = (verts - verts.min()) / (verts.max() - verts.min())

    device = torch.device("cuda:0")
    color1 = torch.tensor([1.0, 0.0, 0.0])
    color2 = torch.tensor([0.0,0.0, 1.0])

    color1 = color1.to(device)
    color2 = color2.to(device)
    color = rgb[:, :, None] * color2 + (1 - rgb[:, :, None]) * color1
    color=color.squeeze(0).permute(1,0,2)
    
    if (type_data != "gt"):
        # verts = verts.unsqueeze(0)
        print("Points shape: ", verts.shape)
        print("RGB shape: ", color.shape)

    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=color)
    num_frames = 36
    render_full = []
    camera_positions = []
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 1.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend[0, ..., :3].detach().cpu().numpy()  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_path, images, duration=12.0, loop=0)    
    print('Done!')

def render_mesh(mesh_src, args, output_file):
    vertices = mesh_src.verts_packed().to(args.device)
    faces = mesh_src.faces_packed().to(args.device)
    color1 = [0.7, 0.0, 0.4]
    color2 = [0.6, 1.0, 1.0]
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    z_min = vertices[:,:,2].min()
    z_max = vertices[:,:,2].max()
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    new_colors = alpha[:, :, None] * torch.tensor(color2).to(args.device) + (1 - alpha[:, :, None]) * torch.tensor(color1).to(args.device)
    textures = pytorch3d.renderer.TexturesVertex(new_colors)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=textures
    )
    renderer = get_mesh_renderer(image_size=512, device=args.device)
    lights = pytorch3d.renderer.PointLights(location=[[0.0, 0.0, -3.0]], device=args.device)
    num_frames = 36
    camera_positions = []
    # output_file = "Results/mesh.gif"
    azim = torch.linspace(0, 360, num_frames)
    for azi in azim:
        azimuth = azi
        distance = 2.0
        elevation = 30.0
        R, T = pytorch3d.renderer.look_at_view_transform(distance, elevation, azimuth, device=args.device, degrees=True)
        camera_positions.append((R,T))
    render_full = []
    for R,T in tqdm(camera_positions):
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=args.device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.detach().cpu().numpy()[0, ..., :3]  # (N, H, W, 3)
        render_full.append(rend)
    images = []
    for i, r in enumerate(render_full):
        image = Image.fromarray((r * 255).astype(np.uint8))
        images.append(np.array(image))
    imageio.mimsave(output_file, images, duration=12.0, loop=0)
    print('Done!')
    


def evaluate_model(args):
    model = SingleViewto3D(args)
    model.to(args.device)
    model.eval()
    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iter1 = checkpoint['step']
        print(f"Succesfully loaded iter {start_iter1}")

    print("Starting evaluating!")

    # Define the mean and standard deviation values specific to your dataset
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Create a white image
    white_image = Image.new('RGB', (224, 224), (255, 255, 255))
    # Create a black image
    black_image = Image.new('RGB', (224, 224), (0, 0, 0))
    # Create a gaussian noise image
    # Define the size of the image
    width = 224
    height = 224

    # Create a numpy array of random values with the same size as the image
    noise = np.random.normal(loc=0.5, scale=0.1, size=(height, width, 3))

    # Convert the numpy array to a PIL image
    noise_image = Image.fromarray((noise * 255).astype(np.uint8))
    

    # Define transformations to preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Preprocess the white image
    white_image_tensor = preprocess(white_image)
    white_image_tensor = white_image_tensor.unsqueeze(0)  # Add batch dimension
    # Preprocess the black image
    black_image_tensor = preprocess(black_image)
    black_image_tensor = black_image_tensor.unsqueeze(0)
    # Preprocess the gaussian noise image
    gaussian_image_tensor = preprocess(noise_image)
    gaussian_image_tensor = gaussian_image_tensor.unsqueeze(0)
    # Ensure white_image_tensor has the same shape as expected by normalize
    white_image_tensor = white_image_tensor.permute(0, 2, 3, 1)  # Change the channel order
    black_image_tensor = black_image_tensor.permute(0, 2, 3, 1)
    gaussian_image_tensor = gaussian_image_tensor.permute(0, 2, 3, 1)
    # Move the input tensor to the same device as the model (GPU)
    white_image_tensor = white_image_tensor.to(args.device)
    black_image_tensor = black_image_tensor.to(args.device)
    gaussian_image_tensor = gaussian_image_tensor.to(args.device)

    # Save white black and gaussian image as png
    white_image.save('Results/white_image.png')
    black_image.save('Results/black_image.png')
    noise_image.save('Results/gaussian_image.png')

    # Manually apply normalization
    white_image_tensor = (white_image_tensor - torch.tensor(mean).to(args.device)) / torch.tensor(std).to(args.device)
    black_image_tensor = (black_image_tensor - torch.tensor(mean).to(args.device)) / torch.tensor(std).to(args.device)
    gaussian_image_tensor = (gaussian_image_tensor - torch.tensor(mean).to(args.device)) / torch.tensor(std).to(args.device)
    # Debugging: print normalized tensor shape
    print("normalized white_image_tensor shape:", white_image_tensor.shape)

    predictions_white = model(white_image_tensor, args)
    predictions_black = model(black_image_tensor, args)
    predictions_gaussian = model(gaussian_image_tensor, args)
    if args.type == "vox":
        predictions_white = torch.sigmoid(predictions_white)
        predictions_black = torch.sigmoid(predictions_black)
        predictions_gaussian = torch.sigmoid(predictions_gaussian)

    if args.type == "vox":
        predictions_white = predictions_white.permute(0, 1, 4, 3, 2)
        predictions_black = predictions_black.permute(0, 1, 4, 3, 2)
        predictions_gaussian = predictions_gaussian.permute(0, 1, 4, 3, 2)
    
    if args.type == "vox":
        render_voxels(predictions_white, output_path=f'Results/Interpret_voxel_white.gif')
        render_voxels(predictions_gaussian, output_path=f'Results/Interpret_voxel_gaussian.gif')
        render_voxels(predictions_black, output_path=f'Results/Interpret_voxel_black.gif')
    elif args.type == "point":
        render_points(predictions_white, output_path=f'Results/Interpret_pcl_white.gif', type_data="pred")
        render_points(predictions_black, output_path=f'Results/Interpret_pcl_black.gif', type_data="pred")
        render_points(predictions_gaussian, output_path=f'Results/Interpret_pcl_gaussian.gif', type_data="pred")
    elif args.type == "mesh":
        render_mesh(predictions_white, args, output_file=f'Results/Interpret_mesh_white.gif')
        render_mesh(predictions_black, args, output_file=f'Results/Interpret_mesh_black.gif')
        render_mesh(predictions_gaussian, args, output_file=f'Results/Interpret_mesh_gaussian.gif')
    
    # plt.imsave(f'Results/{start_iter1}_{args.type}.png', white_image_tensor.squeeze().detach().cpu().numpy())

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
