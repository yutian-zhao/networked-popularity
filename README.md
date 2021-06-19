# YouTube Vevo Music Video Popularity Study

This repository holds our work: **Understanding popularity of online music videos: from a perspective of network structures**, and the code that genrates the plots in our work. The dataset used is [Vevo Music Graph dataset](https://github.com/avalanchesiqi/networked-popularity).

## Modification of original repository
The code is developed based on the forked [repository](https://github.com/avalanchesiqi/networked-popularity) with following updates.
1. Add `mean_plot`, `mean_scatter`, `plot_cumulative` functions to [utils/plot.py](/utils/plot.py).
2. Add [utils/bridges.py](/utils/bridges.py).
3. Add `self.embed_all_genre_dict` to [utils/data_loader.py](/utils/data_loader.py) to load more genre information.


## Code usage
To run the code, download `network_pickle.tar.bz2` and `vevo_en_videos_60k.tar.bz2` from the data host. See [data](#data). 

Extract `network_pickle_{0}.pkl` from `network_pickle.tar.bz2` to 'data/network_pickle' folder. 

Extract `vevo_en_videos_60k.json` from `vevo_en_videos_60k.tar.bz2` to 'data/' folder.

We provide two versions of code to generate plots.
1. [generate_plots.py](/generate_plots.py)
2. [generate_plots.ipynb](/generate_plots.ipynb)

While Jupyter notebook provides an interactive interface, the Python script is more convinient to generate plots in one go. To run the Python scripy, use the following command.
```bash
python generate_plots.py [-h] [--title] [--plots PLOT_ID [PLOT_ID ...]]
```
To generate all plots:
```bash
python generate_plots.py
```
To generate plots with titles:
```bash
python generate_plots.py --title
```
To generate a specific plot (the 2nd plot in the example):
```bash
python generate_plots.py --plots 2
```
To get help message:
```bash
python generate_plots.py -h
```

## Packages dependency
All codes are developed and tested in Python 3.8.8, with following packages.

    # Name          Version
    numpy           1.19.2
    matplotlib      3.3.4
    scipy           1.6.2
    networkx        2.5
    powerlaw        1.4.6
    tqdm            4.59.0 

## Data
The original dataset is hosted on [Google Drive](https://drive.google.com/drive/folders/19R3_2hRMVqlMGELZm47ruk8D9kqJvAmL?usp=sharing) and [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TORICY).
See more details in the [data description](/data/README.md).

## Reference
> [Siqi Wu](https://avalanchesiqi.github.io/), [Marian-Andrei Rizoiu](http://www.rizoiu.eu/), and [Lexing Xie](http://users.cecs.anu.edu.au/~xlx/). Estimating Attention Flow in Online Video Networks. *ACM Conference on Computer-Supported Cooperative Work and Social Computing (CSCW)*, 2019. \[[paper](https://avalanchesiqi.github.io/files/cscw2019network.pdf)\|[slides](https://avalanchesiqi.github.io/files/cscw2019slides.pdf)\|[blog](https://medium.com/acm-cscw/how-does-the-network-of-youtube-music-videos-drive-attention-42130144b59b)\]
