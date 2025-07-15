# Adapted from: https://github.com/PaulJaulhiac/niftiConcrete/blob/6b566bfc7ff9cb7ea7aae3e7536d713d16fec34e/convertKey2Vtk.py

import os
import pandas as pd
import vtk
from scipy.spatial.distance import cdist
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_key_file(key_file_path):
    """
    Reads the key file and returns the data as a DataFrame.
    Expects the file to have 13 columns (after skipping 6 header rows) with the following order:
    X, Y, Z, Scale, o11, o12, o13, o21, o22, o23, o31, o32, o33, e1, e2, e3, InfoFlag
    """
    try:
        data = pd.read_csv(key_file_path, skiprows=6, header=None, sep=r'\s+', usecols=range(17))
        data.columns = ['X', 'Y', 'Z', 'Scale', 
                        'o11', 'o12', 'o13', 'o21', 'o22', 'o23', 'o31', 'o32', 'o33', 'e1', 'e2', 'e3', 'InfoFlag']
        # Check for duplicate positions
        duplicates = data.duplicated(subset=['X', 'Y', 'Z'], keep=False)
        if duplicates.any():
            logging.warning(f"Duplicate positions found in {key_file_path}: {data[duplicates]}")
        return data
    except Exception as e:
        logging.error(f"Error reading key file {key_file_path}: {e}")
        raise

def check_close_spheres(data, threshold=0.1):
    """
    Identifies spheres that are closer than a specified threshold.
    """
    coordinates = data[['X', 'Y', 'Z']]
    distances = cdist(coordinates, coordinates)
    close_pairs = (distances < threshold) & (distances > 0)
    # if close_pairs.any():
    #     # logging.info("Close spheres detected:")
    #     for i, j in zip(*close_pairs.nonzero()):
    #         if i < j:
    #             logging.info(f"Spheres at index {i} and {j} are {distances[i, j]:.3f} units apart.")
    return close_pairs

def get_orientation_transform(row, index):
    """
    Generates a vtkTransform for orientation vectors based on the row data.
    """
    transform = vtk.vtkTransform()
    if index == 0:
        transform.RotateWXYZ(90, row['o11'], row['o12'], row['o13'])
    elif index == 1:
        transform.RotateWXYZ(90, row['o21'], row['o22'], row['o23'])
    elif index == 2:
        transform.RotateWXYZ(90, row['o31'], row['o32'], row['o33'])
    return transform

def create_vtk_from_data(data, vtk_file_path, output_type='both'):
    """
    Generates a VTP file from key file data.
    output_type can be:
      'both'    - include both spheres and arrows,
      'spheres' - only spheres,
      'arrows'  - only arrows.
    """
    os.makedirs(os.path.dirname(vtk_file_path), exist_ok=True)
    append_filter = vtk.vtkAppendPolyData()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    
    positive_count = 0
    negative_count = 0
    for idx, row in data.iterrows():
        if output_type in ['both', 'spheres']:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(row['Scale'])
            sphere.SetCenter(row['X'], row['Y'], row['Z'])
            sphere.SetThetaResolution(30)
            sphere.SetPhiResolution(30)
            sphere.Update()
            append_filter.AddInputData(sphere.GetOutput())
            
            num_cells = sphere.GetOutput().GetNumberOfCells()
            # Color assignment based on InfoFlag, if available.
            if 'InfoFlag' in data.columns:
                if row['InfoFlag'] == 0:
                    sphere_color = [0, 0, 255]
                    positive_count += 1
                elif row['InfoFlag'] == 16:
                    sphere_color = [255, 255, 0]
                    negative_count += 1
                else:
                    sphere_color = [255, 255, 255]
            else:
                sphere_color = [255, 255, 255]
            
            for _ in range(num_cells):
                colors.InsertNextTuple3(*sphere_color)
                
        if output_type in ['both', 'arrows']:
            # Create arrows using orientation vectors
            for i in range(3):
                arrow = vtk.vtkArrowSource()
                arrow.SetTipLength(0.3)
                arrow.SetTipRadius(0.05)
                transform = vtk.vtkTransform()
                transform.Translate(row['X'], row['Y'], row['Z'])
                transform.Scale(10, 10, 10)
                transform.Concatenate(get_orientation_transform(row, i))
                transform_filter = vtk.vtkTransformPolyDataFilter()
                transform_filter.SetTransform(transform)
                transform_filter.SetInputConnection(arrow.GetOutputPort())
                transform_filter.Update()
                append_filter.AddInputData(transform_filter.GetOutput())
                num_cells = transform_filter.GetOutput().GetNumberOfCells()
                color_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
                arrow_color = color_palette[i % 3]
                for _ in range(num_cells):
                    colors.InsertNextTuple3(*arrow_color)
    
    append_filter.Update()
    append_filter.GetOutput().GetCellData().SetScalars(colors)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_file_path)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()
    
    print(f"Number of positive features: {positive_count}")
    print(f"Number of negative features: {negative_count}")

def create_vtk_for_positive_negative(data, vtk_file_path):
    """
    Generates a VTP file highlighting positive (blue) and negative (yellow) features based on the InfoFlag.
    Positive features: InfoFlag == 0 (blue)
    Negative features: InfoFlag == 16 (yellow)
    Also calculates the combined volume of these features (assuming 'Scale' is the radius).
    
    Returns:
        A dictionary with keys: 'positive_volume', 'negative_volume', 'total_volume'
    """
    append_filter = vtk.vtkAppendPolyData()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")
    
    positive_count = 0
    negative_count = 0
    for idx, row in data.iterrows():
        if row['InfoFlag'] == 16:
            negative_count += 1
            sphere_color = [255, 255, 0]  # Yellow
        elif row['InfoFlag'] == 0:
            positive_count += 1
            sphere_color = [0, 0, 255]  # Blue
        else:
            continue
        
        sphere_source = vtk.vtkSphereSource()
        sphere_source.SetRadius(row['Scale'])
        sphere_source.SetCenter(row['X'], row['Y'], row['Z'])
        sphere_source.SetThetaResolution(30)
        sphere_source.SetPhiResolution(30)
        sphere_source.Update()
        append_filter.AddInputData(sphere_source.GetOutput())
        
        for _ in range(sphere_source.GetOutput().GetNumberOfCells()):
            colors.InsertNextTuple3(*sphere_color)
    
    append_filter.Update()
    append_filter.GetOutput().GetCellData().SetScalars(colors)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtk_file_path)
    writer.SetInputData(append_filter.GetOutput())
    writer.Write()
    
    # print(f"Number of positive features: {positive_count}")
    print(f"Number of negative features: {negative_count}")
    
    # Calculate volumes assuming 'Scale' is the radius (V = 4/3 * pi * r^3)
    total_positive_volume = 0.0
    total_negative_volume = 0.0
    for idx, row in data.iterrows():
        if row['InfoFlag'] == 0:
            total_positive_volume += (4.0/3.0)*np.pi*(row['Scale']**3)
        elif row['InfoFlag'] == 32:
            total_negative_volume += (4.0/3.0)*np.pi*(row['Scale']**3)
    # total_volume = total_positive_volume + total_negative_volume
    
    print(f"Negative volume: {total_negative_volume:.3f}")
    
    return {
        'negative_volume': total_negative_volume,
    }

def main():
    base_path = os.path.expanduser('/home/aryanshukla/Desktop/Experiments/Crystals/data')
    key_dir = os.path.join(base_path, 'key_website_pre-compiled')
    vtk_dir_base = os.path.join(base_path, 'vtk_website_pre-compiled')
    
    # List to store volume data for each molecule
    volume_data = []
    
    # Process every .key file in the key directory
    for key_file in os.listdir(key_dir):
        if key_file.endswith('.key'):
            key_file_path = os.path.join(key_dir, key_file)
            molecule_name = os.path.splitext(key_file)[0]
            molecule_vtk_dir = os.path.join(vtk_dir_base, molecule_name)
            os.makedirs(molecule_vtk_dir, exist_ok=True)
            vtk_file_path_base = os.path.join(molecule_vtk_dir, molecule_name)
            
            if os.path.exists(key_file_path):
                data = read_key_file(key_file_path)
                
                # Generate VTK files for various representations
                create_vtk_from_data(data, f"{vtk_file_path_base}_combined.vtp", output_type='both')
                create_vtk_from_data(data, f"{vtk_file_path_base}_spheres.vtp", output_type='spheres')
                create_vtk_from_data(data, f"{vtk_file_path_base}_arrows.vtp", output_type='arrows')
                volumes = create_vtk_for_positive_negative(data, f"{vtk_file_path_base}_positive_negative.vtp")
                
                volume_data.append({
                    'Molecule': molecule_name,
                    'NegativeVolume': volumes['negative_volume']
                })
            else:
                logging.error(f"Key file not found: {key_file_path}")
    
    # Write volume data to a CSV file
    if volume_data:
        df_volumes = pd.DataFrame(volume_data)
        csv_file_path = os.path.join(base_path, 'volume_summary.csv')
        df_volumes.to_csv(csv_file_path, index=False)
        logging.info(f"Volume summary written to {csv_file_path}")
    else:
        logging.info("No volume data to write.")

if __name__ == "__main__":
    main()
