# https://github.com/cdalvaro/machine-learning-master-thesis

```console
thesis/thesis.tex:the OpenClust catalogue~\cite{dias2002new} (Figure~\ref{fig:OpenClustComplete})
thesis/thesis.tex:  \includegraphics[width=0.9\textwidth]{../figures/openclust_catalogue.png}
thesis/thesis.tex:  \caption{OpenClust Catalogue Distribution}
thesis/thesis.tex:  \label{fig:OpenClustComplete}
thesis/thesis.tex:As shown in Figure~\ref{fig:OpenClustSelection},
thesis/thesis.tex:  \caption{OpenClust Catalogue Selection Distribution}
thesis/thesis.tex:  \label{fig:OpenClustSelection}
thesis/thesis.tex:This script is prepared to load the OpenClust catalogue, connect to the Gaia DR2 database,
thesis/thesis.tex:           OpenClust catalogue multiplied by 1.5 (as explained in~\ref{sec:data_mining})}
thesis/references.bib:  url       = {https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/openclust.html}
README.md:the [OPENCLUST][openclust] <span id="a3">[[3]](#f3)</span> catalogue has been used to restrict
README.md:  <img src="https://github.com/cdalvaro/machine-learning-master-thesis/raw/main/figures/openclust_catalogue.svg" title="OpenClust Catalogue Distribution" heigh="256px">
README.md:  <figcaption align=center>OpenClust Catalogue Distribution</figcaption>
README.md:3. <span id="f3"></span> W. Dias, B. Alessi, A. Moitinho, and J. Lépine. New catalogue of optically visible open clusters and candidates. _Astronomy & Astrophysics_, 389(3):871–873, 2002. URL https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/openclust.html. [↩️](#a3)
README.md:[openclust]: https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/openclust.html
paper/paper.tex:Our model contributes with several novelties: i) it is \emph{non-supervised} and \emph{non-parameterized}, making easier the automation process of analyzing a wide range of regions with different typologies, and ii) it is computationally efficient to run in common workstations because it has been developed with Python using the Keras framework which takes advantage of modern GPUs to perform its computations. That increases significantly the computational capacity of our model doing it possible to run on regular workstations.
paper/paper.tex:We start by selecting a region from the OpenClust~\cite{dias2002new} catalog and downloading it from the Gaia DR2 database. The radius of the downloaded region from Gaia is 1.5 times greater than that recorded in OpenClust, ensuring that it includes several stars that do not belong to the open cluster.
paper/paper.tex:Our DECOCC model was implemented in Python 3.8 using the Keras 2.2 framework and Jupyter Notebooks~\cite{Kluyver2016jupyter}. All tests were run on an Apple Mac Pro Late 2013 with a 2.7GHz 12-Core Intel Xeon E5-2697v2, and 64GB RAM 1866MHz DDR3. For the GPU, we use a graphic card AMD FirePro D700 with 6GB~\footnote{All resources developed for this project are available at \href{https://github.com/cdalvaro/machine-learning-master-thesis}{https://github.com/cdalvaro/machine-learning-master-thesis}}.
paper/paper.tex:This paper presents a model to characterize open clusters, comprising an Artificial Neural Network, which is neither parameterized nor supervised. The model does not require complex hardware. Instead, it can be run on workstations with a common GPU, making it accessible for deployment in many research centers.
paper/references.bib:  url       = {https://heasarc.gsfc.nasa.gov/W3Browse/star-catalog/openclust.html}

```
