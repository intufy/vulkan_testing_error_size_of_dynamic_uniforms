Tutorial Samples
================
A set of samples to illustrate Vulkan API on Android with Android Studio		
To build on windows for tutorial02/03, copy/install ndk-r12 ( or better ) to a directory close to root dir ( C: ) to workaround command path 260 character limit issue; see tutorial02/03's build.gradle for details. To Build samples:
- Inside Andriod, menu File --> New --> Import Project
- Find your sample's build.gradle and open it
This will force andrioid studio to create local.properties file to config your SDK and NDK location; if you see error during importing, open local.properties file and correct the NDK and SDK location, then re-sync.

Other Resources:	
Additional Android Studio/NDK samples:    
- [Android Vulkan API Basic Samples](https://github.com/googlesamples/vulkan-basic-samples)
- [Google Play Game Samples](https://github.com/playgameservices/cpp-android-basic-samples)
- [Google Android NDK Samples](https://github.com/googlesamples/android-ndk)

Contents
-------------
- [Tutorial 1 - Load Vulkan](./tutorial01_load_vulkan)
  - create a vulkan device
- [Tutorial 2 - Prebuilt Layers](./tutorial02_prebuild_layers)
  - create a vulkan device with vulkan validation layers
- [Tutorial 3 - Traceable Layers](./tutorial03_traceable_layers)
  - create a vulkan device with validation layers that could be traced.
Refer to README.md under its directory
- [Tutorial 4 - First Window](./tutorial04_first_window)
  - create a vulkan window with WSI 
- [Tutoiral 5 - Triangle](./tutorial05_triangle)
  - draw a simple triangle with Android Studio integrated shaderc feature
- [Tutorial 6 - Texture](./tutorial06_texture)
  - draw textured triangle with [CDep](https://github.com/google/cdep) packed shaderc pre-built binary
  - glsl shaders are compiled at app run time (similar to openGL traditional shader compiling model)

Pre-requisites
--------------
- A device running Android 7.0 (API level 24) or higher
- [Android Studio 3.0](https://developer.android.com/studio/index.html) or higher
- Android NDK
    * [NDK](https://developer.android.com/ndk/downloads/index.html)
    * SDK with the latest cmake package

Test Matrix
------------
| Andrid Studio Version | cmake in SDK| NDK      | device  |
|-----------------------|-------------|----------|---------|
| 3.0.0                 | 3.6.4111459 | NDK-r16  | Pixel XL|

Known Issue:

Getting Started
---------------
- From the Welcome screen, select "Import Project" (or, if you already have a project open, select File > New > Import Project)
- Navigate to the sample you downloaded and click OK

License
-------
Copyright 2016 Google, Inc.

Licensed to the Apache Software Foundation (ASF) under one or more contributor
license agreements.  See the NOTICE file distributed with this work for
additional information regarding copyright ownership.  The ASF licenses this
file to you under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License.  You may obtain a copy of
the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
License for the specific language governing permissions and limitations under
the License.
