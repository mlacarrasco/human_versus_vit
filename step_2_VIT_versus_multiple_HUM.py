import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
import scipy
import scikit_posthocs as sp
import seaborn as sns


def replace_zero(np_arr, small_number=1e-10):
    
    # Find the indices where the values are 0
    zero_indices = np_arr == 0

    # Replace 0 values with a small number greater than 0
    np_arr[zero_indices] = small_number

    # Return the updated array
    return np_arr

def kullback(dir_obj,heatmap_detail):
  array = [dir_obj]

  #leemos cada uno de los promedios por objeto según cada sigma
  hum = np.load(f'resultados_ViT/{dir_obj}/hum/cum_reminder/npy_cum_reminder_{heatmap_detail}.npy')
  p = hum
  p_f = replace_zero(p)
  p_f = cv2.normalize(p_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
  p_f =  p_f/sum(p_f.flatten()) #suma 1

  #en la carpeta GMM se encuentran los datos de cada una de las cabezas de ViT
  npy_folder = f'resultados_ViT/{dir_obj}/gmm'
  npy_files = sorted([file for file in os.listdir(npy_folder) if file.endswith('.npy')])

  # la comparación la hacemos entre un objeto promedio de cada usuario 
  # versus cada una de las cabezas
  for npy_file in npy_files:
    print(f'\tnpy gmm [{npy_file}]')
    vit_gmm = np.load(f'resultados_ViT/{dir_obj}/gmm/{npy_file}')
    q = vit_gmm
    q_f = replace_zero(q)
    q_f = cv2.normalize(q_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
    q_f =  q_f/sum(q_f.flatten()) #suma 1

    if q_f.shape!=p_f.shape:
      try:
        new_dim = np.array(q_f.shape)[::-1]
        p_f = cv2.resize(p_f, new_dim,cv2.INTER_CUBIC)
        p_f = cv2.normalize(p_f, None, 0.0, 1.0, cv2.NORM_MINMAX)
        p_f =  p_f/sum(p_f.flatten()) #suma 1
        print(f'shape qf:{q_f.shape}, pf:{p_f.shape}')
        

      except:
        print('Error: no es posible hacer broadcasting')

    #kullb_hum_to_vit_et = scipy.stats.entropy(p_f, q_f)
    
    #kullb_hum_to_vit_et = scipy.special.rel_entr(p_f, q_f)    
    kullb_hum_to_vit_et = scipy.special.kl_div(p_f, q_f)    
    kullb_hum_to_vit_et[kullb_hum_to_vit_et==np.inf]=0
    kl = np.sum(kullb_hum_to_vit_et)
    array.append(kl)

  return(array)

def step_1(num, heatmap_detail):
  
    if num+1<10:
        fil_end = f'0{num+1}'
    else:
        fil_end = num+1

    filename = f'data_frame_{fil_end}.pkl'
    lista = [
      'cesteria_01', 'cesteria_02', 'cesteria_03', 'cesteria_04', 'cesteria_05',
      'cesteria_06', 'cesteria_07', 'cesteria_08', 'cesteria_09', 'cesteria_10',
      'jarra_01', 'jarra_02', 'jarra_03', 'jarra_04', 'jarra_05',
      'jarra_06', 'jarra_07', 'jarra_08', 'jarra_09', 'jarra_10',
      ]
 
  
    col = ['name_img', 'avg' ,'0', '1', '10', '11', '2' , '3' ,'4', '5', '6' , '7', '8', '9'
      ]

  
    for obj in lista:
        print(f'processing: {obj}...')
        col = np.vstack([col,kullback(obj,heatmap_detail)])

    columns = col[0,:]
  
    df = pd.DataFrame(col[1:,:], columns=columns)

  
  
    df.to_pickle(filename)
    print(df)
    print(f'{filename} has been created...\n\n') 
    return df

# ////////////////////////////  
#           MAIN  
# ////////////////////////////  


if __name__=='__main__':
    # este código permite extraer la distancia entre cada parámetro de Sigma
    # y cada uno de los objetos. NO realiza los gráficos.
    
    rango = np.linspace(0.1, 3.0, 30).round(1)
    for num,heatmap_detail in enumerate(rango):
        print(f'> heatmap parameter: {heatmap_detail}\n\n')
        step_1(num, heatmap_detail)
        print(f'\nfinished....{heatmap_detail}')

    print(f'\nfinished....all')