# Build snpehelper
## 1. Enter Admin Mode
Open a terminal and switch to admin mode:
```
su
oelinux
```
2. Remove Old Build and Compile
Run the following commands to remove the old build, create a new build folder, and compile:
```
rm -r build
mkdir build
cd build
cmake ..
make
```
This will generate libsnpehelper.so.

4. Move libsnpehelper.so to the Tutorials Folder
```
mv libsnpehelper.so ../Tutorials/
```
Now, libsnpehelper.so is ready for use in the Tutorials folder. 🚀








You said:
give me readme format for example # nenerate snpehelper ....


ChatGPT said:
Here’s a properly formatted README.md file for your instructions:

markdown
Copy
Edit
# Generate snpehelper

## Modify Dependencies Path in CMakeLists.txt

1. Open `CMakeLists.txt`.
2. Locate **Line 12**:
   ```cmake
   set(SNPE_ROOT "/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828")
Update the SNPE_ROOT path according to your SNPE installation path.

Build snpehelper
1. Enter Admin Mode
Open a terminal and switch to admin mode:

bash
Copy
Edit
su
oelinux
2. Remove Old Build and Compile
Run the following commands to remove the old build, create a new build folder, and compile:

bash
Copy
Edit
rm -r build
mkdir build
cd build
cmake ..
make
This will generate libsnpehelper.so.

3. Move libsnpehelper.so to the Tutorials Folder
bash
Copy
Edit
mv libsnpehelper.so ../Tutorials/
Now, libsnpehelper.so is ready for use in the Tutorials folder. 🚀

pgsql
Copy
Edit

This format follows **Markdown best practices** for a README file, making it easy to read and copy-paste. Let me know if you need further modifications!






