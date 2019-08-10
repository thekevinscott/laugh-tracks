# Laughter from a Can

![Spam](https://i.ytimg.com/vi/6mvkhXyIL-4/maxresdefault.jpg)

This is a project to teach a machine to provide canned laugh tracks to normal, everyday life. The goal is to use Machine Learning to listen to an audio stream and identify when the optimal time is to insert some laughter.

## Background & Tools

The project is separated into multiple phases, aimed at building increasing levels of understanding.

### Phase One

First, we need to generate a corpus of laughter.

[VGGish]() is a pretrained model for identifying audio.



## Getting Started

There is a [Dockerfile](/somewhere) to make it easy to spin up. Build and run with:

```
docker build -t laugh-tracks .
nvidia-docker run -it --rm -v $(pwd):/notebooks/ -p 8889:8888  --name laugh-tracks laugh-tracks
```

or in fish shell:

```
docker build -t laugh-tracks .
nvidia-docker run -it --rm -v (pwd):/notebooks/ -p 8889:8888  --name laugh-tracks laugh-tracks
```
