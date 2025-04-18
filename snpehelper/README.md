# Build snpehelper
## 1. Enter Admin Mode
Open a terminal and switch to admin mode:
```
su
oelinux
```
## 2. Remove Old Build and Compile
Run the following commands to remove the old build, create a new build folder, and compile:
```
rm -r build
mkdir build
cd build
cmake ..
make
```
This will generate libsnpehelper.so in build folder.

## 3. Move libsnpehelper.so to the Tutorials Folder
```
mv libsnpehelper.so ../Tutorials/
```
Now, libsnpehelper.so is ready for use in the Tutorials folder. 🚀
