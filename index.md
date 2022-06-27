# GSORNet

## Introduction

## Data

### Introduction

There have been a range of impressive datasets for spacial reasoning models, including datasets using static image for directional relationships (CLEVR), videos for temporal and causal reasoning (CLEVERER), and videos including long term temporal and containment reasoning (CATER). Our diagnostic dataset is most similar to CLEVR, using static images with a fixed set of shapes, sizes, and materials. It includes ground truth information about the objects in each scene and the relative directional relationships between those objects (left, right, in front, behind). Our dataset improves on the previous examples by including new more complex ways for objects to be oriented in the scenes and more ground truth information for the relationships between objects. In addition to being placed “independently” on the table, objects can be contained within eachother, or supported on top of one another. In addition, we include the ground truth information for the possible relationships between objects; we store information about if objects could be contained or supported by other objects in the scene based on their relative sizes and shapes. The contain and support predicates are defined and discussed more below.

### Definitions

There are guidelines for how objects can be oriented within the scene depending on their size and shape. We came up with three distinct terms to describe the different methods of placing objects: independent, contained, and supported. Any object can be placed independently, meaning the object is placed on the table not touching any other objects. The contained and supported relationships are defined below:

**Contained:**
an object (containee) is able to be contained by another object (container) when more than half of the bounding box of the containee is able to fit inside of the bounding box of the container such that the container and containee are not intersecting. An equivalent definition would say that the center point of the bounding box of the containee must fit inside of the bounding box of the container. 

**Supported:**
 an object (supportee) can be supported by another object (supporter) if the cross section of the supporter’s bounding box along the table’s plane can fully contain the cross section of the supportee’s bounding box.

### Generation 

Our scene generation process is very similar to that of CLEVR. To generate a scene, we start with a hand-designed default scene containing 6 cameras, a flat gray plane to act as the table, and lighting. Each scene uses the same base table, cameras, and lights, but for each new scene, random jitter is added to the camera and light positions.

For each scene a random number between three and six objects is chosen. For each new object to add, a distinct combination of size, shape, material, and color are chosen. Then based on the size and shape of the new object, we create two lists: the existing objects in the scene which can act as supporters or containers, respectively, for the new object. It’s only possible for an object to have a single “child” containee or supportee, so if any object is already containing or supporting a child object, then it’s not appended to the list and not considered as a candidate. Next, the method of placement is randomly chosen for the object: independent, contained, or supported. If the chosen method is independent, then we choose a random position and orientation until the new object is not in collision with any existing objects. If the method is supported or contained, then a “parent” supporter or container is randomly chosen and the object is placed accordingly. 

Once all objects have been placed in a scene, the ground truth predicates are generated and the scene is saved. 

![GeoSpa Shapes](images/geospa-shapes.PNG)

### Predicates

There are a total of 8 predicates included in the ground truth data: left, right, in front, behind, contains, supports, can contain, and can support. The left, right, in front, and behind predicates are generated the same as CLEVR and indicate the directional relationships between object pairs.  The contains and supports predicates indicate the current orientation of objects within the scene based on the placement type (independent, contained, supported). 

The can-contain and can-support predicates represent the potential for objects to fit contained inside of or supported on top of other objects based on the scene, including considering any objects already contained or supported. For both the can-contain and can-support relationships, object A can be contained/supported by object B if object A can be moved to fulfil the definition of the contain/support relationship (defined above) without disturbing any other objects in the scene. Some potential cases if object A can be contained/supported in object B are outlined below.

*Object A already a supporter.*
If object A is already supporting another object C, then it cannot be contained/supported by any other object B since object C would likely fall off. 

*Object A already a container.*
If object A is containing another object D, then object D will remain inside of object A even if object A is moved, so we must still consider if object B can contain/support object A. 

*Object B already a container/supporter.*
If object B is already containing/supporting object E, then when considering can contain/support of object A, we consider the size and shape of object E. We do not consider the position of object E i.e., it would be acceptable for object E’s position to shift slightly as we move object A into the relationship in/on object B. 

Definitions of can-contain and can-support predicates:

**Can-contain:**
Its possible for object A to be contained inside of object B such that object A can be moved into object B without disturbing any other objects in the scene. Any objects already contained within object B in the scene must be able to be in object B along with object A.

**Can-support:**
It’s possible for object A to be supported on object B such that object A can be moved onto object B without disturbing any other objects in the scene. Any objects already supported above object B in the scene must be able to be on object B along with object A.

### Distribution

## Model

![SORNet Trnasformer (Embedding) Network](images/sornet-transformer.PNG)
![SORNet Readout Network](images/sornet-readout.PNG)

### Implementation

### Training and Hyperparameter

### Testing and Experiment Setup

## Results

### 1-view Results

![1 View Bar](images/2viewbar.PNG)

#### Qualitative Analysis

![1 View Example 1](images/1view1.PNG)
![1 View Example 2](images/1view2.PNG)

#### Quantitative Analysis

### 2-view Results

#### Qualitative Analysis

![2 View Example 1](images/2view1.PNG)
![2 View Example 2](images/2view2.PNG)
![2 View Example 3](images/2view3.PNG)

#### Quantitative Analysis

![2 View Bar](images/2viewbar.PNG)

### Kitchen Objects

#### Qualitative Analysis

#### Quantitative Analysis

## Conclusion

### Larger Implications

### Future Work

## References
