import colorsys as cs
from more_itertools import roundrobin
import itertools
import random

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.cluster import (HDBSCAN,
                             MeanShift,
                            )
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn import manifold

import pacmap
import umap
import trimap
from classix import CLASSIX

def HLSvaluegradient(colours):
    '''Change RGB-coded colours by random saturation and
    random lightness.
    
    Parameters
    -----------
    colours : Array-like 
        Collection of RGB-coded colours.
    
    Returns
    -------
    List 
        List of  RGB-coded colours with randomised saturation
        and lightness.    
    '''
    
    colours = [cs.rgb_to_hls(*c[:-1]) 
               for c in colours
              ]
    random_light = 0.95 #np.random.uniform(0.5,1.0)
    random_saturation = 0.95 #np.random.uniform(0.667,1.0)
    colours = [[h[0], l*random_light, s*random_saturation] 
               for h, l, s
               in zip(colours,
                      np.linspace(1.00,
                                  0.45,
                                  len(colours),
                                 ),
                      np.linspace(0.575,
                                  1.000,
                                  len(colours),
                                 ),
                     )
              ]
    return [cs.hsv_to_rgb(*c) for c in colours]

def SegGradCMap(n_colors=9,
                steps=3,
                cmap='gist_rainbow'
               ):
    '''Produces a colormap with segmented coulours and 
    their gradual versions. 
    
    Parameters
    -----------
    n_colors : Int, default: 9
        The number of colours to be equidistantly sampled
        from `cmap`
    steps : String, default: 3
        Number of versions for the gradual steps between colours. 
    cmap : String, default: 'gist_rainbow'
        Colormap from which the colours are sampled
    
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
    '''
    colour_segments = np.linspace(0,1,n_colors*steps)
    colour_segments = colour_segments.reshape(n_colors,
                                              steps,
                                             )
    # colour_segments
    third = np.round(len(colour_segments)/3).astype(int)
    colour_segments = list(roundrobin(colour_segments[:third:],
                                      colour_segments[third*2::],
                                      colour_segments[third:third*2:],
                                     )
                          )

    colors = []
    for i_colour, segment in enumerate(colour_segments):
        # newcolours = mpl.cm.gist_rainbow(segment)
        newcolours = mpl.colormaps[cmap](segment)
        newcolours = HLSvaluegradient(newcolours)

        for newcolour in newcolours:
            colors.append(newcolour,
                     )

    label = f'{n_colors:.0f}x{steps:.0f}Colours'

    SegGrad = (mpl
               .colors
               .ListedColormap(name=label,
                               colors=colors,
                              )
              )

    return SegGrad

def EmbeddingClustering_Iter(X,
                             embedding,
                             clustering,
                             df,
                             iterations=200,
                             seed=None,
                            ):

    embedding_key, embedding_params = embedding
    clustering_key, clustering_params = clustering
    # seed = 13

    random.seed(seed)
    np.random.seed(seed)
    dict_embeddings = {'PaCMAP':pacmap.PaCMAP,
                       'LocalMAP':pacmap.LocalMAP,
                       'tSNE':manifold.TSNE,
                       'UMAP':umap.UMAP,
                       'TriMAP':trimap.TRIMAP,
                      }
    clu_minp = 2
    dict_clusterings = {'HDBSCAN': HDBSCAN,
                        'CLASSIX': CLASSIX,
                        'MeanShift': MeanShift,
                       }

    clustering = dict_clusterings[clustering_key]
    clustering = clustering(**clustering_params)
    for i in range(iterations):
        embedding = dict_embeddings[embedding_key]
        if seed and embedding_key != 'TriMAP':
            embedding_params.update(random_state=seed)
        embedding = embedding(**embedding_params)
        X1 = embedding.fit_transform(X)
        X1 = MinMaxScaler().fit_transform(X1)
        labels = (clustering.fit_transform(X1)
                  if clustering_key=='CLASSIX'
                  else clustering.fit_predict(X1)
                 )
        df.loc[:,f'cluster_{i:03.0f}'] = labels
        seed += 1
    return [embedding_key,
            clustering_key,
            df,
           ]

def EmbClu_AMI(EmbClu, df):
    clustering_i = (df
                    .iloc[:, 1:]
                    .columns
                   )
    clusts_combinations = (itertools
                           .combinations(clustering_i,
                                         2,
                                        )
                          )
    # ARIs=[]
    AMIs=[]
    for cs in clusts_combinations:
        AMIs.append(AMI(*df
                        .loc[:,cs]
                        .T
                        .values
                       )
                   )
    return [EmbClu, AMIs]

def ClusterNo_AMI(combination, embeddings_clusterings):
    df_clusters = embeddings_clusterings[combination]
    df_ClNo = pd.DataFrame(Counter([len(set(c))
                                    for c
                                    in df_clusters
                                    .iloc[:, 1:]
                                    .T
                                    .values
                                   ]
                                  ),
                           index=[0],
                          )
    df_ClNo = df_ClNo.reindex(df_ClNo
                              .T
                              .sort_values(by=0,
                                           ascending=False,
                                          )
                              .index,
                              axis=1,
                             ).T
    df_ClNo.rename(columns={0:'counts'},
                   inplace=True,
                  )
    df_ClNo['pct'] = df_ClNo['counts'].apply(lambda x:
                                             (x
                                              /df_ClNo['counts'].sum()
                                             )
                                             *100
                                            )

    clusters_list = sorted(df_ClNo.T.columns)
    CluMet_dict_clust = {}

    for n_clusters in clusters_list:
        n_clusters_cols = [c
                           for c
                           in df_clusters.columns
                           if len(set(df_clusters
                                      [c]
                                      .values
                                     )
                                 )==n_clusters
                          ]

        AMIs=[]
        for cs in itertools.combinations(n_clusters_cols,
                                         2,
                                        ):
            AMIs.append(AMI(*df_clusters
                            .loc[:,cs]
                            .T
                            .values
                           )
                       )
        CluMet_dict_clust[n_clusters] = [AMIs]

    return [combination, df_ClNo, CluMet_dict_clust]
    
def AdjacentValues(vals, q1, q3):
    ''' For violin plotting
    '''
    vals = sorted(vals)
    upper_adjacent_value = (q3
                            + (q3
                               - q1
                              )
                            * 1.5
                           )
    upper_adjacent_value = np.clip(upper_adjacent_value,
                                   q3,
                                   vals[-1]
                                  )

    lower_adjacent_value = (q1
                            - (q3
                               - q1
                              )
                            * 1.5
                           )
    lower_adjacent_value = np.clip(lower_adjacent_value,
                                   vals[0],
                                   q1
                                  )
    return lower_adjacent_value, upper_adjacent_value