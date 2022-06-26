# GSORNet

## Introduction

## Data

### Definition

While creating the data we had to create guidelines for how objects could be oriented within the scene depending on their size and shape. We came up with three distinct terms to describe the different methods of placing objects: independent, contained, and supported. Any object can be placed independently, meaning the object is placed on the table not touching any other objects. The contained and supported relationships are defined below:

Contained:
an object (containee) is able to be contained by another object (container) when more than half of the bounding box of the containee is able to fit inside of the bounding box of the container such that the container and containee are not intersecting. An equivalent definition would say that the center point of the bounding box of the containee must fit inside of the bounding box of the container.

Supported:
an object (supportee) can be supported by another object (supporter) if the cross section of the supporter’s bounding box along the table’s plane can fully contain the cross section of the supportee’s bounding box.


### Predicates

### Generation

To generate a scene, we start with a hand-designed default scene containing 6 cameras, a flat gray plane to act as the table, and lighting. For each scene, random jitter is added to the positions of the cameras and lights. A random number of objects is chosen for the scene.

### Distribution

## Model

### Implementation

### Training and Hyperparameter

### Testing and Experiment Setup

## Results

### 1-view Results

#### Qualitative Analysis

#### Quantitative Analysis

### 2-view Results

#### Qualitative Analysis

#### Quantitative Analysis

### Kitchen Objects

#### Qualitative Analysis

#### Quantitative Analysis

## Conclusion

### Larger Implications

### Future Work

## References
