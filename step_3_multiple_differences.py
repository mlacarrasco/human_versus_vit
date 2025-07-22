import os
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy

plt.rcParams['font.size']=8
plt.rcParams.update({"font.family": 'Helvetica'})

def graficos(num, filename):

    print(f'loading {filename} file...')
    df = pd.read_pickle(filename)


    df = df.apply(pd.to_numeric, errors='ignore')
    #print(df.describe())

    #sns.heatmap(data = df.iloc[:,1:], yticklabels=df.iloc[:,0] )
    #plt.show()

    sprays = df.to_numpy()

    fig = plt.figure(figsize=(12, 4), dpi=350)
    ax = fig.add_subplot(111)
    boxprops = dict(linestyle='-', linewidth=1, color='blue')
    medianprops = dict(linestyle='-', linewidth=2,color='black')
    wiskers = dict(linestyle='--', linewidth=1, color='gray')

    ticks_names = [f'{col}' for col in range(1,13)]
    data_plot = [df[f'{col}'] for col in range(12)]
    
    #separamos por tipo
    data_plot_cesteria = [df[:10][f'{col}'] for col in range(12)]
    data_plot_jarrones = [df[10:][f'{col}'] for col in range(12)]
    
    n_datos = len(data_plot[0])
    n_datos_objeto = len(data_plot_cesteria[0])
    
    #box = sns.violinplot([df["0"], df["1"],df["2"], df["3"],df["4"], df["5"],df["6"], df["7"],df["8"], df["9"],df["10"], df["11"]] , palette="Set3")
    #box = sns.violinplot(data=data_plot)
    ax.boxplot(data_plot,
                    showfliers=False, showmeans=True,
                    boxprops=boxprops,
                    bootstrap=None,
                    medianprops=medianprops,
                    whiskerprops = wiskers)


    for pos_x in range(0,12):
        y_cesteria = data_plot_cesteria[pos_x]
        y_jarron = data_plot_jarrones[pos_x]
        random_x = np.random.normal(0, 0.08, size=n_datos_objeto)
        x = random_x+pos_x+1
        ax.plot(x,y_cesteria, 'ro', alpha=0.2)
        ax.plot(x,y_jarron, 'gs', alpha=0.2)

    #generamos data vacía para la leyenda
    ax.plot([], [], '^', linewidth=1, c='green', label='median')
    ax.plot([], [], 'ro', linewidth=1, c='red', label='Basketry KL distance',alpha=0.2)
    ax.plot([], [], 'gs', linewidth=1, c='green', label='Vase KL distance',alpha=0.2)
    ax.plot([], [], '-', linewidth=2, c='black', label='mean')

    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend()
    ax.get_xticklabels(ticks_names)
    ax.set_ylim([0,2.0])
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f'{num}_box_plot_{filename[:-4]}.png', dpi='figure',  pad_inches='layout')

    #promedio de las diferencias por cabeza. Considerar que cada fila representa los datos de 20 objetos
    promedio = np.mean(data_plot_cesteria, axis=1)

    
    hsd = scipy.stats.tukey_hsd(data_plot[0], 
                                data_plot[1], 
                                data_plot[2], 
                                data_plot[3], 
                                data_plot[4],
                                data_plot[5], 
                                data_plot[6], 
                                data_plot[7], 
                                data_plot[8], 
                                data_plot[9],
                                data_plot[10], 
                                data_plot[11])
    
    print('*'*100)
    print(filename[:-4])
    #print(hsd)
    print('*'*100)
    cof = hsd.confidence_interval(confidence_level=.95)

    p_val = hsd.pvalue

    
   
    etiquetas = []
    indices = set()
    for ((i, j), l) in np.ndenumerate(p_val):
    # filter out self comparisons
        
        if i != j:
            umbral = p_val[i,j]
            if umbral<0.05:
                indices.add(i)
                indices.add(j)
                
                print(f"({i+1} - {j+1}) *{umbral:>6.4f}")

    heads= 12
    labels = np.arange(1,heads+1)
    fig = plt.figure(dpi=300)
    ax = fig.gca()
    sns.heatmap(data = p_val, yticklabels=labels , ax=ax, xticklabels=labels, cmap='RdYlBu_r', fmt= '.4g')

    plt.tight_layout()
    plt.savefig(f'heamap_p_val_{filename[:-4]}.png', dpi='figure',  pad_inches='layout')
    
    print(indices)
    #error estandar (calculamos los intervalos de confianza)
    err_std = np.std(data_plot, axis=1)/np.sqrt(len(data_plot)) * 1.96
    
    
    return np.reshape(promedio,shape=(-1,1)), np.reshape(err_std,shape=(-1,1))
    #plt.show()
    


# ////////////////////////////////////  
#                 MAIN  
# ////////////////////////////////////  


if __name__=='__main__':

    # este código lee cada archivo de distancia y los grafica de acuerdo
    # al valor de sigma.
    pkl_files = sorted([file for file in os.listdir() if file.endswith(".pkl")])
    flag = True
    for num, filename in enumerate(pkl_files):
        if num+1<10:
            init = f'0{num+1}'
        else:
            init = num+1
        
        mediana, std = graficos(init, filename)
        if flag:
            data = mediana
            data_std = std
            flag = False
        else:
            data = np.hstack((data, mediana))
            data_std =  np.hstack((data_std, std))

    #ojo que los datos se deben ver en horizontal
    heads = ['.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'H', 'd', 'P', 'X']
    heads_label = [f'head #{i+1}' for i in range(12)]
    plt.figure()
    table = pd.DataFrame(data.T)
    rows, cols = table.shape
    
    fig, ax = plt.subplots(ncols=1, nrows=1)    
    fig.set_figwidth(12)

    for head_i in range(cols):
        ax.plot(np.arange(rows), 
                table[head_i], 
                marker=heads[head_i], 
                label=heads_label[head_i])
        
    ax.grid(axis='x', alpha=0.1)
    ax.grid(axis='y', alpha=0.2)
    #ax.xticks(np.arange(rows))
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'rendimiento_mediana.png', dpi='figure',  pad_inches='layout')
    

    #otro gráfico
  
    fig, ax = plt.subplots(ncols=3, nrows=4, dpi=350)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    table = pd.DataFrame(data.T)
    table_std = pd.DataFrame(data_std.T)
    rows, cols = table.shape
    
    valores = 20
    sigma_vals = np.linspace(0.1, 3.0, valores).round(1)
    xticks = np.linspace(0,rows-1, len(sigma_vals))
    

    for head_i in range(cols):
        ix,iy = np.unravel_index(head_i, (4, 3))
        ax[ix,iy].text(28,1.10, f'head # {head_i+1}',horizontalalignment='right',
                       bbox = dict(facecolor = 'white', alpha = 0.8),fontsize = 13)
        
        ax[ix,iy].plot(np.arange(rows),table, color='gray', linewidth=0.5, alpha=0.5) 
        pos_min = np.argmin(table[head_i])
        ax[ix,iy].stem(pos_min,table[head_i][pos_min],linefmt='--r')

        ax[ix,iy].plot(np.arange(rows), 
                table[head_i],  
                label=heads_label[head_i], color='red',linewidth=3) 

        ax[ix,iy].fill_between(np.arange(rows),table[head_i]+table_std[head_i],table[head_i]-table_std[head_i], alpha=0.2, color='tab:cyan')
        #ax[ix,iy].fill_between(np.arange(rows),, alpha=0.05, color='tab:cyan')
        #ax[ix,iy].errorbar(np.arange(rows), 
        #        table[head_i], yerr=table_std[head_i], alpha=0.1, color='red') 

        
        ax[ix,iy].grid(axis='x', alpha=0.1)
        ax[ix,iy].grid(axis='y', alpha=0.2)
        #ax[ix,iy].fill_between(np.arange(rows),table[head_i], alpha=0.1, color='tab:brown')

        ax[ix,iy].set_xticks(ticks=xticks, labels=sigma_vals, rotation=90)
        ax[ix,iy].set_ylim((0,1.25))
        #ax[ix,iy].set_xlim(0,np.max(xticks))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'rendimiento_mediana_multiple.png', dpi='figure',  pad_inches='layout')

        



    #otro gráfico : mide la varianza
  
    fig, ax = plt.subplots(ncols=3, nrows=4, dpi=350)
    fig.set_figwidth(12)
    fig.set_figheight(12)
    table = pd.DataFrame(data_std.T)
    rows, cols = table.shape
    
    valores = 30
    sigma_vals = np.linspace(0.1, 3.0, valores).round(1)
    xticks = np.linspace(0,rows-1, len(sigma_vals))
    

    

    for head_i in range(cols):
        ix,iy = np.unravel_index(head_i, (4, 3))
        ax[ix,iy].text(25,0.35, f'head # {head_i+1}',horizontalalignment='right',
                       bbox = dict(facecolor = 'white', alpha = 0.8),fontsize = 13)
        
        ax[ix,iy].plot(np.arange(rows),table, color='gray', linewidth=1, alpha=0.3) 
        pos_min = np.argmin(table[head_i])
        ax[ix,iy].stem(pos_min,table[head_i][pos_min],linefmt='--r')

        ax[ix,iy].plot(np.arange(rows), 
                table[head_i],  
                label=heads_label[head_i], color='red',linewidth=3) 
        
        ax[ix,iy].grid(axis='x', alpha=0.1)
        ax[ix,iy].grid(axis='y', alpha=0.2)
        ax[ix,iy].fill_between(np.arange(rows),table[head_i], alpha=0.1, color='tab:brown')

        ax[ix,iy].set_xticks(ticks=xticks, labels=sigma_vals, rotation=90)
        ax[ix,iy].set_ylim((0,0.4))
        #ax[ix,iy].set_xlim(0,np.max(xticks))
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(f'rendimiento_varianza_multiple.png', dpi='figure',  pad_inches='layout')
