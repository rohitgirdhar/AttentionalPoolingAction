BOOST_DIR := /home/rgirdhar/Software/basic/boost/install2/
BOOST_LIB_DIR := $(BOOST_DIR)/lib
BOOST_INC_DIR := $(BOOST_DIR)/include
TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
LDFLAGS := -Wl,-rpath,$(BOOST_LIB_DIR)  # this ensures it will look in the correct BOOST directory for libs (not the system path)

all: pose_to_heatmap.so zero_out_channels.so render_pose.so render_objects.so

pose_to_heatmap.so: pose_to_heatmap.cc
	g++ -std=c++11 $(LDFLAGS) -shared -I$(BOOST_INC_DIR) `pkg-config --cflags --libs opencv` pose_to_heatmap.cc -o pose_to_heatmap.so -fPIC -I $(TF_INC) -O2 -L$(BOOST_LIB_DIR) -lboost_system -lboost_filesystem -lboost_thread

zero_out_channels.so: zero_out_channels.cc
	g++ -std=c++11 $(LDFLAGS) -shared zero_out_channels.cc -o zero_out_channels.so -fPIC -I $(TF_INC) -O2

render_pose.so: render_pose.cc pose_utils.hpp
	g++ -std=c++11 $(LDFLAGS) -shared -I$(BOOST_INC_DIR) `pkg-config --cflags --libs opencv` render_pose.cc -o render_pose.so -fPIC -I $(TF_INC) -O2 -L$(BOOST_LIB_DIR) -lboost_system -lboost_filesystem -lboost_thread

render_objects.so: render_objects.cc
	g++ -std=c++11 $(LDFLAGS) -shared -I$(BOOST_INC_DIR) `pkg-config --cflags --libs opencv` render_objects.cc -o render_objects.so -fPIC -I $(TF_INC) -O2 -L$(BOOST_LIB_DIR) -lboost_system -lboost_filesystem -lboost_thread
