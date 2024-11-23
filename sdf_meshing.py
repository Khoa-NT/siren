'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
#!/usr/bin/env python3

import logging
import numpy as np
import plyfile
import skimage.measure
import time
import torch


### In test_sdf.py, N=1600
### Khoa: N should > 100 to ensure the coordinates are in range [-1, 1].
### see /home/khoa/workspace/playground/test_diffusion_gan/Diffusers/3D_Liver_Reconstruction/notebook/train_MLP/train_mlp.ipynb
### for more details.
def create_mesh(
    decoder, filename, N=256, max_batch=64 ** 3, offset=None, scale=None
):
    start = time.time()
    ply_filename = filename

    decoder.eval()

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1] ### Define the bottom-left-down corner of the sampling grid
    voxel_size = 2.0 / (N - 1) ### Calculate size of each voxel to span [-1, 1] in each dimension

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor()) ### Create a tensor of indices
    samples = torch.zeros(N ** 3, 4) ### 4th column will store SDF values

    # transform first 3 columns
    # to be the x, y, z index
    ### Khoa: samples is a 3D volumn grid of points with N^3 points and origin at (0, 0, 0).
    ### But this box is centered at (-1, -1, -1) and spans [2, 2, 2].
    samples[:, 2] = overall_index % N ### z coordinate
    samples[:, 1] = (overall_index.long() / N) % N ### y coordinate
    samples[:, 0] = ((overall_index.long() / N) / N) % N ### x coordinate

    # transform first 3 columns
    # to be the x, y, z coordinate
    ### Then convert grid indices to actual 3D coordinates
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]


    ### ------------------------------- Infer SDF values ------------------------------- ###
    ### Maximum number of points in overall_index and samples
    num_samples = N ** 3

    ### Disable gradient computation
    ### but kind of redundant here because torch.zeros has requires_grad=False by default.
    samples.requires_grad = False

    ### Counter
    head = 0

    ### Process points in batches to avoid memory issues
    while head < num_samples:
        print(head)
        ### Move batch to GPU
        ### Khoa: We can create slice() here
        sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

        ### Khoa: and reuse the slice() here
        ### Infer SDF values for the current batch of points
        samples[head : min(head + max_batch, num_samples), 3] = (
            decoder(sample_subset)
            .squeeze()#.squeeze(1) ### Use squeeze() to get a `tensor scalar` instead of using item() returning a `scalar`
            .detach()
            .cpu()
        )
        head += max_batch

    ### Get SDF values
    sdf_values = samples[:, 3]
    sdf_values = sdf_values.reshape(N, N, N)

    end = time.time()
    print("sampling takes: %f" % (end - start))

    ### Convert SDF values to a mesh
    convert_sdf_samples_to_ply(
        sdf_values.data.cpu(),
        voxel_origin,
        voxel_size,
        ply_filename + ".ply",
        offset,
        scale,
    )


def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)

    ### Khoa: Fixed marching_cubes_lewiner --> marching_cubes
    ### https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes
    try:
        # verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    ### marching_cubes() use `lewiner` by default which uses left-handed coordinate system.
    ### Therefore, the results are flipped in x and y axes.
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    ### ---------------------- Exporting with plyfile ---------------------- ###
    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    ### Create 1D array of tuples (x, y, z) coordinates
    ### `f4` means float32
    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :]) ### Each index is a tuple of (x, y, z)

    ### Create 1D array of tuples ([vert_index1, vert_index2, vert_index3],)
    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),))) ### ((faces[i, :].tolist(),)) --> ([vert_index1, vert_index2, vert_index3],)

    ### `i4` means int32
    ### (3,) means a 1D array of length 3, which represents 3 vertex indices for each face
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    logging.debug("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)

    logging.debug(
        "converting to ply format and writing to file took {} s".format(
            time.time() - start_time
        )
    )
