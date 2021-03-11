This source is edited by Intufy(intufy@gmail.com) for testing the error size of dynamic uniforms.
Check out a constant ERROR_COUNT in forTestingDynamincUniform function of VulkanMain.cpp. 
Also you can check out the simple dynamic uniform uses in tri.frag and tri.vert which are shader sources.
The problem is that a screen size triangle is shown when ERROR_COUNT is not muliple of 1024, but nothing is shown when it is a multiple of 1024.
The phenomenon is being appeared on Android devices. 
I didn't test on Windows nor Linux. It works well on all Apple devices including laptops and mobiles.

The original source is https://github.com/googlesamples/android-vulkan-tutorials

