import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np
import os
import cv2

#Versión original: José Aranda
#mod:Miguel Carrasco (29/05/2025)


# funcion que crea distribucion gaussiana atencion, guarda csv regular y ajustada a vit
def create_heatmap(surface_df,csv_file, cover_img, i, dir_obj, heatmap_detail,user_id, to_show):

    cover_img = plt.imread(cover_img)
    gaze_on_surf = pd.read_csv(surface_df)

    #gaze_on_surf = surface_df
    #gaze_on_surf = surface_df[surface_df.on_surf == True]
    #gaze_on_surf = surface_df[(surface_df.confidence > 0.8)]
    dim_ima = np.array(cover_img.shape[0:2])
    grid = dim_ima//20 # height, width of the loaded image
    

    #heatmap_detail = 0.02 #0.01 # this will determine the gaussian blur kerner of the image (higher number = more blur)

    #print(grid)

    gaze_on_surf_x = gaze_on_surf['x_norm']
    gaze_on_surf_y = gaze_on_surf['y_norm']

    # flip the fixation points
    # from the original coordinate system,
    # where the origin is at botton left,
    # to the image coordinate system,
    # where the origin is at top left
    gaze_on_surf_y = 1 - gaze_on_surf_y

    # make the histogram
    hist, x_edges, y_edges = np.histogram2d(
        gaze_on_surf_y,
        gaze_on_surf_x,
        range=[[0, 1.0], [0, 1.0]],
        #normed=True,
        bins=grid
    )
    #added--->
    hist = cv2.resize(hist,dim_ima[::-1], interpolation=cv2.INTER_CUBIC )
    # gaussian blur kernel as a function of grid/surface size
    filter_h = int(heatmap_detail * grid[0]) // 2 * 2 + 1
    filter_w = int(heatmap_detail * grid[1]) // 2 * 2 + 1
    heatmap = gaussian_filter(hist, sigma=(filter_w, filter_h), order=0)

    #print(heatmap)
    #print(type(heatmap))
    # Specify the file path and name
    #file_path = 'heatmap.csv'
    # Save the array to a CSV file
    #np.savetxt(file_path, heatmap, delimiter=',')


    # Step 1: Get height and width
    height, width = heatmap.shape

    # Step 2: Calculate remainder
    height_remainder = height % 16
    width_remainder = width % 16

    # Step 3: Eliminate last rows and columns
    new_height = height - height_remainder
    new_width = width - width_remainder

    # Update the array by keeping only the relevant portion
    your_array = heatmap[:new_height, :new_width]


    # Save the array to a CSV file
    np.save(f'resultados_ViT/{dir_obj}/hum/npy_reminder_heatmap_{csv_file[:-4]}_{dir_obj}.npy', your_array)

    if to_show: 
      plt.figure(figsize=(8,8))
      plt.imshow(cover_img)
      plt.imshow(your_array, cmap='jet', alpha=0.5)
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(f"resultados_ViT/{dir_obj}/hum/user_{user_id}_reminder_heatmap_{heatmap_detail}.png")
      
      plt.show()

    #np.savetxt(f"resultados_ViT/{dir_obj}/hum/reminder_heatmap_{csv_file[:-4]}_{dir_obj}.csv", your_array, delimiter=',')

#///////////////////////////////////////////////////////////////////////////
def avg_heatmap(dir_obj, jpg_file, heatmap_detail, to_show):
# Initialize an empty DataFrame to store the cumulative sum of values
    cumulative_sum = None
    #print("avg")
    # Specify the path to the folder containing CSV files
    csv_folder = f"resultados_ViT/{dir_obj}/hum/"
    #csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv")])
    npy_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".npy")])

    for npy_file in npy_files:
      
        npy_path = os.path.join(csv_folder, npy_file)
        print(f'reading {npy_file}...')
        if cumulative_sum is None:
          #cumulative_sum =  pd.read_csv(csv_path, header=None)
          cumulative_sum = np.load(npy_path)
        else:
          #cumulative_sum.add(pd.read_csv(csv_path, header=None))
          cumulative_sum += np.load(npy_path)

    cumulative_sum = cumulative_sum / len(npy_files)
  

    # Optionally, save the result to a new CSV file
    #average_values.to_csv('average_values_reminder.csv', index=False)
    #cumulative_sum.to_csv(f"resultados_ViT/{dir_obj}/hum/cum_reminder.csv", index=False, header=None)
    np.save(f"resultados_ViT/{dir_obj}/hum/cum_reminder/npy_cum_reminder_{heatmap_detail}.npy",cumulative_sum)

    cover_img = plt.imread(jpg_file)
    
    if to_show: 
      plt.figure(figsize=(8,8))
      plt.imshow(cover_img)
      plt.imshow(cumulative_sum, cmap='jet', alpha=0.5)
      plt.axis('off')
      plt.tight_layout()
      plt.savefig(f"{jpg_file[:-4]}_cum_heatmap_20_{heatmap_detail}.png")  
     
      plt.show()
    
    
    return()

#///////////////////////////////////////////////////////////////////////////
def gmm(list,heatmap_detail):
  for object in list:
    csv_folder = f"resultados_ViT/{object}/pupil_data/"


    # List files in both CSV and JPG folders
    csv_files = sorted([file for file in os.listdir(csv_folder) if file.endswith(".csv")])
    jpg_file = f"resultados_ViT/imagenes_vit/{object}.jpg"

    # Process each pair of CSV and JPG files

    for user_id, csv_file in enumerate(csv_files):
      print(f'processing: [{csv_file}]....')
      csv_path = os.path.join(csv_folder, csv_file)

      #creacion csv de dits guassiana
      create_heatmap(csv_path,csv_file, jpg_file, jpg_file[:-4], object, heatmap_detail, user_id, to_show=False)

    #print(csv_file)
    #print(jpg_file)
    print(f'creating average image for {object}\n\n')
    avg_heatmap(object,jpg_file, heatmap_detail, to_show=False)


# ////////////////////////////////////  
#                 MAIN  
# ////////////////////////////////////  

if __name__=='__main__':
  # este código calcula la imagen promedio de los usuarios. Cada promedio por objeto 
  # se almacena en un archivo .npy
  
  list = [
    "cesteria_01", "cesteria_02", "cesteria_03", "cesteria_04", "cesteria_05",
    "cesteria_06", "cesteria_07", "cesteria_08", "cesteria_09", "cesteria_10",
    "jarra_01", "jarra_02", "jarra_03", "jarra_04", "jarra_05",
    "jarra_06", "jarra_07", "jarra_08", "jarra_09", "jarra_10",
  ]

  
  # se modifica el valor de sigma para generar un mapa promedio según sigma
  rango = np.linspace(0.1, 3.0, 30).round(1)
  for heatmap_detail in rango:
    print(f'> heatmap parameter: {heatmap_detail}\n\n')
    gmm(list,heatmap_detail )
    print(f'\nfinished....{heatmap_detail}')
