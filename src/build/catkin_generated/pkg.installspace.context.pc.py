# generated from catkin/cmake/template/pkg.context.pc.in
CATKIN_PACKAGE_PREFIX = ""
PROJECT_PKG_CONFIG_INCLUDE_DIRS = "${prefix}/include".split(';') if "${prefix}/include" != "" else []
PROJECT_CATKIN_DEPENDS = "roscpp;rospy;geometry_msgs;sensor_msgs;std_msgs;dynamic_reconfigure".replace(';', ' ')
PKG_CONFIG_LIBRARIES_WITH_PREFIX = "-lcv_all".split(';') if "-lcv_all" != "" else []
PROJECT_NAME = "cv_all"
PROJECT_SPACE_DIR = "/usr/local"
PROJECT_VERSION = "0.0.1"
