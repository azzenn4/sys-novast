
# DEVELOPER


| Name                  | Affiliation       | Role             | Contact                   |
|-----------------------|-------------------|------------------|---------------------------|
| Rezky Mulia Kam       | BINUS University  | Maintainer       | rezky.kam@binus.ac.id     |
| Juanita G. Bakara     | BINUS University  | Documentation    | juanita.bakara@binus.ac.id|




# PROTOTYPE PHASE





    This source code is currently in development and has not been fully tested for reliability
    in production environments. Use it at your own discretion and with caution. **Read Further 
    Notice at the End**

    System-NovaAstra is developed for adequate scalability with no need to use Google Colab or
    other Cloud Computing services. It can run locally on your Laptop or Desktop, provided
    the hardware meets the requirements.


    The system has been optimized and tested using the "LOWEST MINIMUM REQUIREMENT"
    specifications (detailed below). For reference, initial hardware specification can be
    found in the "DEVELOPMENT HARDWARE" section.

    Linux is the recommended operating system, as kernel parameter optimization is crucial
    and Linux offers superior I/O performance (eg. Tesseract-OCR)

    Developer's Vision:
    
    Initial goal of this project is provided as an Open-Source software specifically designed for the
    mental health industry. It is not intended for revenue generation or profit.
    
    Thus, developer sees an opportunity that this project have a potential for alot more diverse use.





# ENVIRONMENT 





    DRIVER VERSION: 560.35.03
    CUDA VERSION: 12.6
    CUDA TOOLKIT: Cuda compilation tools, release 12.6, V12.6.85
    PYTHON: 3.12.3
    LIBRARY: (INSTALL PIP **IMPORTANT** USE 'pyenv' DON'T USE ANACONDA)
    RUN : pip install -r requirements.txt
    FOR LINUX OPERATING SYSTEM:

    - Set Kernel Parameter:
    vm.swappiness = 1
    vm.dirty_ratio = 10
    vm.dirty_background_ratio = 5 ~ Minimal Impact





    ~ THIS SOFTWARE IS OPTIMIZED FOR USE ON ADA LOVELACE, HOPPER AND AMPERE GPU ARCHITECTURES.
    ~ THIS SOFTWARE IMPLEMENTS CONCURRENT COMPUTING,
    ~ ENSURE: CPU > 4 CORE > 8 THREAD (RECOMMENDED) | INTEL / AMD. Does not support Apple M-Series
    ~ ENSURE: GPU Uses CUDA CORE (Pytorch Library, Flash Attention (Credit: Tri Dao, Stanford University))
    ~ AMD ROCM not tested yet...


>   #  DEVELOPMENT HARDWARE:
    OPERATION SYSTEM: ARCH-LINUX 
    ARCH : x86_64
    KERNEL: 6.12.9-arch-1-1
    CPU: Ryzen 7 7745HX 8 CORES 16 THREADS
    GPU: NVIDIA RTX 4070 Max-Q 5.888 CUDA CORE 8GB
    RAM: 5200MhZ 8x2GB


>   # LOWEST MINIMUM REQUIREMENT:
    OPERATION SYSTEM: UBUNTU 24.04.1 LTS 
    ARCH : x86_64 
    KERNEL: 6.8.0-50-generic
    CPU: INTEL i3-12100F 4 CORES 8 THREADS
    GPU: RTX 2050 2.048 CUDA CORES 4GB
    RAM: 3200MhZ 1x8GB
    Exclusion : Disable ParlerTTS, 4GB VRAM is not enough for Acoustic Modelling.

 

> For use on Windows 11 Operating System:
> Please check the 'to/your/path' format as it's different from Linux






> Due to the need of XAI for this project, we cannot yet ensure its reliability in real world,
> therefore, deploying such tool in production environment is highly not Recommended until further development.
