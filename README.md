# ToonDecompose

The project ToonDecompose is conducted by S2PR to find a way to decompose a cartoon animation into several components (or called "Sprites" in terminology). 

The **inputs** should be 

* Frame images (from an animation video clip).
* Number of wanted components/objects/sprites.

The **outputs** are

* All components' contents (sequential images).
* All components' transforms (sequential homography matrices).

Note that we do NOT want to require users to give any masks of objects, and the entire processing can be fully automatic (unless users want to indicate some instructions, in some special use cases).

Besides, this project use [RAFT](https://github.com/princeton-vl/RAFT) to compute optical flows. Since the optical flow is the only introduced external model prior, trying other models like [FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) are definitely some other options.

We have also published an academic paper (in SIGGRAPH Asia 2022) during the conducting of this project (See also the Citation section).

# Installation

You will need a python 3 environment of Pytorch with CUDA enabled.

    git clone https://github.com/lllyasviel/ToonDecompose.git
    cd ToonDecompose
    pip install -r requirements.txt

For Pytorch, I am using

    torch==1.7.1+cu110
    torchvision==0.8.2+cu110

But I have also tried different versions and many versions should work.

# Hello Violet

In the folder "data\violet", we include a sequence of 78 frames (000.png ~ 077.png), sampled from the animation ["Violet Evergarden"](https://www.youtube.com/results?search_query=Violet+Evergarden).

A screenshot is like this:

![i1](github_page/i1.png)

This Violet example is a good input because

* It is small. Downloading is fast.
* It is real. This Violet is from a REAL animation in the real productions, NOT some toy examples synthesized in some university labs. *(Violet Evergarden is generally considered one of KyoAni's peaks, and is the most well-produced anime between 2015 and 2022.)*
* It is challenging. The motion is very large. The girl is running. Even latest optical flow methods and correspondence-based decomposition methods would fail in this example.

Because of these, we use this example as a typical test.

## Configuration

In "config.py", line 8, we can edit the codes to target a directory for frames

    task_name = 'violet'
    input_path = './data/' + task_name + '/'

The "task_name" can be edited to read different examples, like your own inputs. We highly encourage to first use the default "task_name" to see whether the codes work as expected.

Besides, the number of objects is specified at line 39:

    number_of_objects = 2

By default it is 2.

## Step 0: Compute Optical Flows

Run this to start this step

    python step_0_optical_flow.py

If this works properly, you should see the optical flows computed in "workers\violet\flow".

![i2](github_page/i2.png)

Note that the direction of the flows are visualized with the normal ball (different from mainstream visualizations that use hue wheels). The advantage is that the scale of visualization is invariant across different flow instances.

![i2](github_page/n.png)

Ideally, we should compute all flows from N frames to N frames, resulting in N*N flow instances, but that requires too much computation power. In most cases, using a window of 6 frame yields good enough results.

To be specific, we compute 4 flows for each frame at the step of 2, like:

    Flow from frame N to frame N - 3
    Flow from frame N to frame N - 1
    Flow from frame N to frame N + 1
    Flow from frame N to frame N + 3

For example,

    ...
    Flow from frame 9 to frame 6 (stored as "workers\violet\flow\9T6.npy")
    Flow from frame 9 to frame 8 (stored as "workers\violet\flow\9T8.npy")
    Flow from frame 9 to frame 10 (stored as "workers\violet\flow\9T10.npy")
    Flow from frame 9 to frame 12 (stored as "workers\violet\flow\9T12.npy")
    Flow from frame 10 to frame 7 (stored as "workers\violet\flow\10T7.npy")
    Flow from frame 10 to frame 9 (stored as "workers\violet\flow\10T9.npy")
    Flow from frame 10 to frame 11 (stored as "workers\violet\flow\10T11.npy")
    Flow from frame 10 to frame 13 (stored as "workers\violet\flow\10T13.npy")
    ...

Each flow has a "npy" file of original flow and a "png" image of visualization.

## Step 1: Warmup Coarse Object Alpha

Run this to start this step

    python step_1_warmup_alpha.py

In this step, we will warmup the alpha of each object with a method to analyze the "flow inside/outside each objects". A more detailed explanation can be found the section 3.2 of our SA2022 paper (see also the Citation section).

![i3](github_page/i3.png)

After the computation (about 40 minutes on Nvidia GTX 3070), you should be able to see the alpha masks in "workers\violet\preview_labels":

![i4](github_page/i4.png)

## Step 2: Warmup Coarse (Homography Matrices) Transforms & Step 3: Preview the Transforms

Run this to start this step

    python step_2_warmup_h.py
    python step_3_preview_warmup.py

In this step we initialize a sequence of homography matrix for each component sprite by approximating the previous optical flows.

If the script works properly, you should see the a video that visualizing the transforms in "workers\violet\preview_position.mp4"

![i4](github_page/position.gif)

A MP4 file of the example output can be download [here](github_page/position.mp4).

Note that we requires not some "similar" outputs on your device - we want your outputs to be EXACTLY THE SAME.

If you cannot reproduce the exactly same mp4 output at this step, you are not likely to achieve same outputs in the future steps, since all future optimizations are sensitive to initial parameters.

If you cannot get the same output at this stage, you can open a GitHub Issue so that I can take a look at your case.

My tests on different devices are:

    GTX 3070 - OK
    GTX 3070 Ti - OK
    GTX 3070 TI Laptop GPU - OK
    GTX 980m - Not exactly same but still works with similar outputs
    CPU mode without GPU - Failed
    GTX 1050 - Failed
    GTX 3050 - Failed
    GTX 1660 - Not exactly same but still works, outputs are not very similar

As you can see, the output is subject to some strange effects due to the gpu specifications. We currently do not know what is causing this.

## Step 4: Learning Sprites

Run this to start this step

    python step_4_sprite_learning_s1.py
    python step_5_sprite_learning_s2.py

At this stage we actually optimize the sprites jointly. Some methods involved in this optimization is introduced in that SA2022 paper (see also the Citation section). Note that since I am still actively working on this project, the scripts may have some updates over time (see also the TODO section).

## Step 6: Output the Sprites/Components

Run this to start this step

    python step_6_video.py

This will output all sprites/components in the folder "workers\violet\vis"

![i5](github_page/i5.png)

To be specific, for the frame N, we will have

    N.img.png - the original frame image
    N.homo.png - the visualized final homography 
    N.obj.K.png - the N-th frame of the K-th sprite/component

Also, a mp4 file will be generated at "workers\violet\vis.mp4"

![i5](github_page/i6.gif)

Then you can use those results to some other applications! 

# Citation

The following paper describes several methods used in this project. 

    @Article{ZhangSA2022,
       author    = {Lvmin Zhang and Tien-Tsin Wong and Yuxin Liu},
       title     = {Sprite-from-Sprite: Cartoon Animation Decomposition with Self-supervised Sprite Estimation},
       journal   = "Transactions on Graphics (SIGGRAPH Asia 2022)",
       year      = 2022,
       volume    = 31,
       number    = 1,
    }

The above paper has a [project page]().

# TODO List

* refactor the code and increase readability.
* make a separated branch for that SA2022 paper and include codes that corresponds more precisely to the paper.
* include some download links of more example inputs (with copyrights).

# 中文社区

我们有一个除了技术什么东西都聊的以技术交流为主的群。如果你一次加群失败，可以多次尝试: 816096787。

