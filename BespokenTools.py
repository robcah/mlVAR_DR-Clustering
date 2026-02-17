import colorsys as cs
from more_itertools import roundrobin
import itertools
import random

from collections import Counter
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patheffects import Stroke
from sklearn.cluster import (HDBSCAN,
                             MeanShift,
                            )
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn import manifold
import networkx as nx

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

def SpectrumHSL(segments=11,
             name='spectrumHSL',
             hue_range=[0.0,
                        1.0,
                       ],
             sat_range=[0.67,
                        0.75,
                       ],
             lig_range=[0.5,
                         0.9,
                        ],
            ):
    '''Produces a colormap attempting to minimise the
    green region to have a more diverse colour palette,
    as well as a progressive reduction of the lightness, to
    differentiate the beginning and the end of the
    colormap. 
    
    Parameters
    -----------
    segments : Int, Default : 11
        Segments in which the hue space will be divided
    name : String, Default : 'spectrum'
        Name for the new colormap. 
    hue_range : Array-like, default: [0.0,1.0]
        Range of hue from which the new colormap
        will be sampled
    sat_range : Array-like, default: [0.67,0.75]
        Range of saturation from which the new
        colormap will be sampled
    lig_dange: Array-like, default: [0.5,0.9]
        Range of lightness from which the new 
        colormap will be sampled
    
    
    Returns
    -------
    `matplotlib.colors.ListedColormap`
    '''
            
    colors = mpl.colors
    lspace = np.linspace
    rgb_list = [cs
                .hls_to_rgb(h,l,s)
                             for h,s,l
                             in zip(lspace(*hue_range,
                                           segments,
                                          ),                                         
                                    lspace(*sat_range,
                                           segments
                                          ),
                                    lspace(*lig_range,
                                           segments
                                          ), 

                                   )
                            ]

    rgb_list = np.delete(rgb_list,
                         (segments//2)-1,
                         axis=0,
                        )
    spectrum = (colors
                .LinearSegmentedColormap
                .from_list(name=name,
                           # colors=mpl.cm.hsv(hsv_list),
                           colors=rgb_list,
                          )
               )

    return spectrum

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

def NodeColours(node_list,
                cmap,
                ):
    if not isinstance(node_list, list):
        node_list = list(node_list)
    node_list = sorted(node_list)
    return {n:cmap(i
                   / len(node_list)
                   )
            for i, n
            in enumerate(node_list)
           }

def MinMaxNorm(data,
               range=(0.0,1.0),
              ):
    return (MinMaxScaler(feature_range=range)
            .fit_transform(np
                           .reshape(data,
                                    (-1,1),
                                   )
                          )
            .ravel()
           )

def StandardNorm(data,
               # range=(0.0,1.0),
              ):
    return (StandardScaler()
            .fit_transform(np
                           .reshape(data,
                                    (-1,1),
                                   )
                          )
            .ravel()
           )

def ModifyNodesNames(g,
                     k,
                     v,
                    ):
    if 'DiGraph' not in str(g):
        h = nx.Graph()
    else:
        h = nx.DiGraph()

    h.add_nodes_from([(n
                       .replace(k, v),
                       d)
                      for n, d
                      in sorted(g
                                .nodes(data=True)
                               )
                     ]
                    )
    h.add_edges_from([(n0
                       .replace(k, v),
                       n1
                       .replace(k, v),
                       d)
                      for n0, n1, d
                      in g.edges(data=True)
                     ]
                    )
    return h

def EdgeWeightsPctil(g,
                     value='similarity',
                     top_pct=50,
                    ):

    dict_edges = {(n0,n1):[np.abs(s)]
                  for n0, n1, s
                  in (g
                      .edges
                      .data(value)
                     )
                  if n0 != n1
                 }

    df = (pd
          .DataFrame(dict_edges)
          .T
         )
    pctil = (df
             .sum()
             * (1
                - (top_pct
                   / 100)
               )
            )
    df = (df
          .sort_values(by=0,
                      )
          .reset_index()
         )
    cumsum_col = 'cumsum'
    df[cumsum_col] = df.iloc[:,2].cumsum()
    idx = df[cumsum_col].searchsorted(pctil)[0]
    edges_pctil = (df
                   .iloc[idx:,:]
                   [::-1]
                   .set_index(['level_0',
                               'level_1',
                              ],
                             )
                   .iloc[:,0]
                   .to_dict()
                   .keys()
                  )
    # print(edges_pctil)
    return list(edges_pctil)

def GraphPlot(g,
              g_type,
              title,
              axes,
              i,
              cmap,
              top_pct=80,
             ):
    '''
    Plotting of networks based on a Networkx Graph object.

    Parameters
    ----------
    g: graph
       A networkx graph.
    g_type: str
       Type of graph, either directed or indirected.
    title: str
       Title of the plot
    axes: list of matplotlib axes
        In case of mutiple networks in a plot.
    i: int
        index for the location of the plot in a
        list of axes.
    cmap: matplotlib.Colormap
        Colormap for nodes
    top_pct: int
            Integer between 0 and 100 limiting the
            edges to plot to a specified top
            percentile.

    Returns
    -------
    matplotlib plot of a graph.
    '''
    MMS_width = (MinMaxScaler(feature_range=(0.0,
                                             20.0,
                                            )
                             )
                 .fit_transform
                )
    MMS_alpha = (MinMaxScaler(feature_range=(0.1,
                                             0.8,
                                            )
                             )
                 .fit_transform
                )
    MMS_size = (MinMaxScaler(feature_range=(8_000,
                                            16_000,
                                            )
                             )
                 .fit_transform
                )
    print(g)
    if 'DiGraph' not in str(g):
        arrowstyle='-'
    else:
        arrowstyle=('-|>,head_length=0.5,'
                    'head_width=0.25,'
                    'angleA=90,'
                    'angleB=90,'
                    'widthA=0.5,'
                    'lengthA=0.5,'
                    'scaleA=0.5,'
                    'widthB=0.2,'
                    'lengthB=2.0,'
                    'scaleB=0.2'
                   )
    ax = axes[i]
    ax.set_aspect(1)
    ax.axis("off")
    ax.margins(0.125)
    pos = nx.circular_layout(g)

    dict_edges = {(n0,n1):s
                 for n0, n1, s
                 in (g
                     .edges
                     .data('similarity')
                    )
                }
    edge_similarity = np.fromiter(dict_edges.values(),
                                    dtype=float
                                   )
    edges_pctil = EdgeWeightsPctil(g,
                                   top_pct=top_pct,
                                  )
    loops = {(n0,n1):v
             for (n0,n1), v
             in dict_edges.items()
             if n0==n1
            }
    nonloops = list(zip(*{(n0,n1):v
                     for (n0,n1), v
                     in dict_edges.items()
                     if (n0,n1) not in loops.keys()
                    }.items()
                  ))
    loops = list(zip(*loops.items()))
    dict_width = {}
    dict_alpha = {}
    for items in [nonloops, loops]:
        if len(items) != 0:
            keys, values = items
            values = np.fromiter(values,
                                 dtype=float
                                )
            values = np.abs(values)
            edge_width = (MMS_width(np
                                    .reshape(values,
                                             (-1,1),
                                            )
                                   )
                          .flatten()
                         )
            [dict_width.update({k:v})
             for k,v
             in zip(keys, edge_width)
            ]
            edge_alpha = (MMS_alpha(np
                                    .reshape(values,
                                             (-1,1),
                                            )
                                   )
                          .flatten()
                         )
            [dict_alpha.update({k:v})
             for k,v
             in zip(keys, edge_alpha)
            ]

    edge_width = list({k:dict_width[k]
                  for k
                  in dict_edges.keys()
                 }.values())
    edge_alpha = {k:(dict_alpha[k]
                     if k in edges_pctil
                     or k[0]==k[1]
                     else 0.0
                    )
                  for k
                  in dict_edges.keys()
                 }
    edge_alpha = list(edge_alpha
                     .values()
                     )

    size_values = [v
                   for _, v
                   in (g
                       .nodes
                       .data('strength')
                      )
                  ]
    node_size = (MMS_size(np
                          .reshape(size_values,
                                   (-1,1),
                                  )
                         )
                 .flatten()
                )
    node_color = [cmap(i/len(list(g)))
                  for i, n
                  in enumerate(list(g))
                 ]

    connectionstyle=f'arc3,rad=-0.125'

    edge_color = [[0.0]*3 if s > 0
                  else [0.5,0.0,0.0]
                  for s
                  in edge_similarity
                 ]
    nx.draw_networkx(g,
                     pos=pos,
                     node_size=node_size,
                     nodelist=list(g),
                     with_labels=True,
                     node_color=[1.0]*4,
                     linewidths=8.0,
                     edgecolors=node_color,
                     width=edge_width,
                     edge_color=[c+[a]
                                 for c,a
                                 in zip(edge_color,
                                        edge_alpha,
                                       )
                                ],
                     arrows=True,
                     arrowstyle=arrowstyle,
                     connectionstyle=connectionstyle,
                     ax=ax,
                    )
    [patch
     .set_path_effects([Stroke(joinstyle='miter',
                               capstyle='round',
                              )
                       ]
                      )
     for patch in ax.patches
    ]

    node_dict = {k:v
             for k,v
             in zip(list(g),
                    node_size
                   )
            }

    [t.set_fontsize((node_dict[t.get_text()]
                       ** 0.5
                       )
                       * 1.8
                      / len(t.get_text()
                            .split('\n')
                            [-1]
                           )
                     )
     for t in ax.texts
     if t.get_text() in list(g)
    ]

    ax.set_title(title,
                 fontsize=25,
                 x=0.5,
                 y=1.0,
                 va='bottom',
                 ha='center',
                 transform=ax.transAxes,
                )
    if g_type == 'temporal':
        loops = {n0:i
                 for i, (n0, n1)
                 in enumerate(g.edges())
                 if n0==n1
                }
        node_size = {n:s
                     for n,s
                     in zip(list(g),
                            node_size,
                           )
                    }
        for node, index in loops.items():
            loop = ax.patches[index]
            alpha = loop.get_edgecolor()[-1]
            loop.set_color([0.1,0.1,0.667,alpha])
            if node_size[node]>5_000:
                t2 = (mpl
                      .transforms
                      .Affine2D()
                      .translate(0,0.167)
                      + ax.transData
                     )
                loop.set_transform(t2)
    ax.set_gid(f'id_{i}')

def adjacent_values(vals, q1, q3):
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

def DisplayCentralities(g,
                        caption,
                        sortby='strength',
                        ascending=False,
                        drop='color',
                        head=None,
                        folder_to_save=False,
                       ):
    '''Extraction of Centralities from a
    networkx graph object as a pd.DataFrame.
    '''

    df = (pd
          .DataFrame(dict(g
                          .nodes
                          .data()
                         )
                    )
          .drop(drop,
                axis=0,
               )
          .T
         )
    columns = df.columns
    index= df.index
    stdsc = (StandardScaler()
             .fit_transform(df)
            )
    df = (pd.
          DataFrame(columns=columns,
                    index=index,
                    data=stdsc,
                   )
         )
    table = (df
            .astype(float)
            .sort_values(sortby,
                         ascending=ascending,
                        )
            .head(head)
            .style
            .format(precision=2)
            .background_gradient(cmap='RdYlGn')
            .set_caption(caption)
            )
    display(table)
    if folder_to_save:
        table.to_html(join(in_folder
                           .replace('Data',
                                    'Plots',
                                   ),
                           f'{caption}_centralities_results.html'
                          )
                     )

def DisplayWeights(g,
                   caption,
                   sortby='absolute_similarity',
                   ascending=False,
                   drop='color',
                   head=None,
                   top_pct=None,
                   folder_to_save=False,
                  ):
    '''Extraction of Weights from a networkx 
    graph object as a pd.DataFrame.
    '''
    df = (pd
          .DataFrame({(n0,n1):v
                      for n0,n1,v
                      in g.edges.data()
                     }
                    )
          .T
         )
    (df
     .index
     .set_names(['n0', 'n1'],
                inplace=True,
               )
    )
    df = (df
          .reset_index()
         )
    if not top_pct:
        df = (df
              .sort_values(sortby,
                           ascending=ascending,
                          )
             )
    else:
        pctil = (df
                 [sortby]
                 .sum()
                 * (1
                    - (top_pct
                       / 100)
                   )
                )
        (df
         .sort_values(sortby,
                      inplace=True,
                     )
        )
        cumsum_col = f'{sortby}_cumsum'
        df[cumsum_col] = df[sortby].cumsum()
        idx = df[cumsum_col].searchsorted(pctil)
        df = (df
              .drop(cumsum_col,
                    axis=1,
                   )
              .iloc[idx:,:]
              [::-1]
             )
    df = df.head(head)
    table = (df
            .reset_index(drop=True)
            .style
            .format(precision=2)
            .background_gradient(cmap='RdYlGn')
            .set_caption(caption)
           )
    display(table)
    if folder_to_save:
        table.to_html(join(in_folder
                           .replace('Data',
                                    'Plots',
                                   ),
                           f'{caption}_weights_results.html'
                          )
                     )

# def MosaicPlots(network_list):
    # '''Automatic mosaic layout to plot
    # Strength centrality by node and network.
    # '''
    # n = len(network_list)
    # title_ax = ('AA'*n)[:-1]
    # barsh_layout = '.'.join([str(i)
                             # for i
                             # in range(n)
                            # ]
                           # )
    # mosaic = f'''
              # {title_ax}
              # {barsh_layout}
              # '''
    # return mosaic
def MosaicPlots(network_list,
                first=True,
               ):
    '''Automatic mosaic layout to plot
    Strength centrality by node and network.
    '''
    if any([isinstance(e, str) for e in network_list]):
        cluster_list = range(len(network_list))
    else:
        cluster_list = list(network_list)
    n = len(cluster_list)
    title_ax = ('AA'*n)[:-1] if first else ''
    barsh_layout = '.'.join([str(i)
                             for i
                             in cluster_list
                            ]
                           )
    mosaic = f'''
              {title_ax}
              {barsh_layout}
              '''
    return mosaic
    

def MosaicNetworks(list_):
    n = len(list_)
    rows, residual = (n//4, n%4)
    mosaic = (np
              .arange(4 * rows)
              .reshape(rows, 4)
              .tolist()
             )
    if residual != 0:
        rows += 1
        mosaic.append(['.'] * 8)
        mosaic[:-1] = [sorted(r*2)
                       for r
                       in mosaic[:-1]
                      ]
        edge = 4-residual
        (mosaic[-1]
         [edge:-edge]) = sorted([r
                                 + (4
                                    * (rows-1)
                                   )
                                 for r
                                 in range(residual)
                                ]
                                * 2
                               )
    return mosaic

def RainViolin(df_plot,
               ax,
               dict_colours,
               title,
               fs=12,
               violin=False,
               violin_args_prime=dict(points=200,
                                      vert=False,
                                      widths=1.2,
                                      bw_method=0.333,
                                      alpha=0.75,
                                      linewidth=0.25,
                                      zorder=100,
                                      median_args=dict(marker='d',
                                                       s=60,
                                                       label='median',
                                                       alpha=0.75,
                                                      ),
                                      mean_args=dict(marker='o',
                                                     s=40,
                                                     label='mean',
                                                     alpha=0.75,
                                                    ),
                                      medmean_args=dict(color='white',
                                                        edgecolors='k',
                                                        linewidths=1.0,
                                                        zorder=300,
                                                       ),
                                      hlines_args=dict(color='k',
                                                       linestyle='-',
                                                       alpha=0.5,
                                                       zorder=200,
                                                      ),
                                     ),
               rain_args_prime=dict(s=0.001,
                                    alpha=1.0,
                                    rasterized=False,
                                    zorder=150,
                                    y_jitter=0.1,
                                    x_jitter=0.0333,
                                   )
              ):
    violin_args = violin_args_prime.copy()
    rain_args = rain_args_prime.copy()

    (alpha,
     linewidth,
     zorder,
     median_args,
     mean_args,
     medmean_args,
     hlines_args,
    ) = [violin_args.pop(key) for key in ['alpha',
                                          'linewidth',
                                          'zorder',
                                          'median_args',
                                          'mean_args',
                                          'medmean_args',
                                          'hlines_args',
                                         ]
        ]

    (y_jitter,
     x_jitter,
    ) = [rain_args.pop(key) for key in ['y_jitter',
                                        'x_jitter',
                                         ]
        ]

    parts = ax.violinplot([c.dropna().values
                           for _, c
                           in df_plot.items()
                          ],
                          showmeans=False,
                          showextrema=False,
                          showmedians=False,
                          **violin_args
                         )
    for pc, label in zip(parts['bodies'],
                         df_plot.columns,
                        ):
        pc.set_facecolor(dict_colours[label])
        pc.set_edgecolor(dict_colours[label])
        pc.set_linewidth(0.25)
        pc.set_alpha(0.75)
        pc.set_zorder(100)
    if not violin:
        for j, (pc,
                label,
               ) in enumerate(zip(parts
                                  ['bodies'],
                                  df_plot
                                  .columns,
                                 )
                             ):
            (pc
             .get_paths()[0]
             .vertices[:, 1]) = np.clip(pc
                                        .get_paths()[0]
                                        .vertices[:, 1],
                                        j+1,
                                        j+2,
                                       )

        for i, (label,
                features,
               ) in enumerate(df_plot
                              .T
                              .iterrows()
                             ):
            y = np.full(len(features),
                        i
                        +0.8
                       ).astype(float)
            idxs = np.arange(len(y))
            y.flat[idxs] += (np
                             .random
                             .normal(loc=0.0,
                                     scale=y_jitter,
                                     size=len(idxs),
                                    )
                            )
            features += (np
                         .random
                         .normal(loc=0.0,
                                 scale=x_jitter,
                                 size=len(features),
                                )
                        )
            ax.scatter(features,
                       y,
                       c=dict_colours[label],
                       **rain_args,
                      )

    (quartile1,
     medians,
     quartile3,
    ) = np.nanpercentile(df_plot
                         .T
                         .values,
                         [25, 50, 75],
                         axis=1,
                        )
    means = df_plot.mean()
    whiskers = np.array([adjacent_values(sorted_array,
                                         q1,
                                         q3,
                                        )
                         for sorted_array, q1, q3
                         in zip(df_plot.T.values,
                                quartile1,
                                quartile3,
                               )
                        ]
                       )
    (whiskers_min,
     whiskers_max,
    ) = (whiskers[:, 0],
         whiskers[:, 1],
        )

    inds = np.arange(1,
                     len(medians)+1,
                    )
    median = ax.scatter(medians,
                        inds,
                        **median_args,
                        **medmean_args,
                       )
    mean = ax.scatter(means,
                      inds,
                      **mean_args,
                      **medmean_args,
                     )
    intq50 = ax.hlines(inds,
                       quartile1,
                       quartile3,
                       lw=6.0,
                       label='IQR',
                       **hlines_args,
                      )
    whiskers = ax.hlines(inds,
                         whiskers_min,
                         whiskers_max,
                         lw=3.0,
                         label='IQR x 1.5',
                         **hlines_args,
                        )
    ax.grid(True,
            linestyle=':',
            zorder=1,
           )
    ax.set_title(title,
                 x=0.5,
                 y=1.0,
                 va='bottom',
                 ha='center',
                 fontsize=fs,
                )

    yticks_labels = [f'{l.replace('_',' ')}'
                     for l
                     in df_plot.columns
                    ]

    ax.set_yticks(np.arange(1,
                            len(yticks_labels)
                            + 1
                           ),
                  labels=yticks_labels,
                  fontsize=fs,
                  )
    ax.set_xticks(np.arange(1,8),
                  labels=np.arange(1,8),
                  rotation=45,
                  fontsize=fs,
                 )

def GraphBuilding(nodes, 
                  edges,
                  centralities_weight,
                  directional=False,
                 ):
    if directional:
        g = nx.DiGraph()
    else:
        g = nx.Graph()
    g.add_nodes_from(node for node in nodes.items())
    g.add_edges_from([(c0, 
                       c1,
                       d,
                      )
                      for (c0, c1), d 
                      in edges.items()
                     ],
                    )
    betweenness = nx.betweenness_centrality(g,
                                            endpoints=True,
                                            weight=centralities_weight,
                                           )
    closeness = nx.closeness_centrality(g,
                                        distance=centralities_weight,
                                        wf_improved=True,
                                       )

    strength = {node:StrengthCentrality(g,
                                        node,
                                        centralities_weight,
                                       )
                for node
                in g.nodes()
               }
    [nx.set_node_attributes(g, weight, name)
     for name, weight
     in [['betweenness', betweenness],
         ['strength', strength],
         ['closeness', closeness],
        ]
    ]                    
    return g

def StrengthCentrality(G, node, attribute):
    """
    Node strength is defined as the sum of the strengths (or weights) of all
    of the nodes edges.

    Parameters
    ----------
    G: graph
       A networkx graph
    node: object
       A node in the networkx graph
    attribute: object
       The edge attribute used to quantify node strength

    Returns
    -------
    node strength: (int, float)
    """
    output = 0.0
    for edge in G.edges(node):
        output += np.abs(G.get_edge_data(*edge)[attribute])
    return output