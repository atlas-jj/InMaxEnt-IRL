cmake_minimum_required(VERSION 2.8.3)
project(img2str)

find_package(catkin REQUIRED)

catkin_package()
catkin_python_setup()

install(PROGRAMS 
  main.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
