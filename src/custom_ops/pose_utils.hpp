#include <opencv2/opencv.hpp>

#include <tuple>
#include <iostream>
#include <sstream>

// Very important to add the following #define
// boost json parser depends on boost::spirit
// which is not thread safe by default.
// It was giving Segmentation Faults.
// Also, this means I need to compile with -lboost_thread
// ref: http://stackoverflow.com/a/22089792/1492614
// This was tested to work fine with multi-threaded training
#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using namespace std;
using namespace cv;
namespace pt = boost::property_tree;


vector<float> joint_color {1, 0, 0,
                           1, 0.33, 0,
                           1, 0.66, 0,
                           1, 1, 0,
                           0.66, 1, 0,
                           0.33, 1, 0,
                           0, 1, 0,
                           0, 1, 0.33,
                           0, 1, 0.66,
                           0, 1, 1,
                           0, 0.66, 1,
                           0, 0.33, 1,
                           0, 0, 1,
                           0.33, 0, 1,
                           0.66, 0, 1,
                           1, 0, 1,
                           1, 0, 0.66,
                           1, 0, 0.33};
//                           1, 1, 1};
vector<int> limbSeq {2, 3,
                     2, 6,
                     3, 4,
                     4, 5,
                     6, 7,
                     7, 8,
                     2, 9,
                     9, 10,
                     10, 11,
                     2, 12,
                     12, 13,
                     13, 14,
                     2, 1,
                     1, 15,
                     15, 17,
                     1, 16,
                     16, 18,
                     3, 17};
//                     6, 18};

#define RENDER_POSE_OUT_TYPE_RGB 1
#define RENDER_POSE_OUT_TYPE_SPLITCHANNEL 2
Mat render_pose(vector<vector<tuple<float,float,float>>> poses,
    int out_ht, int out_wd,
    int max_ht, int max_wd, int marker_wd,
    int out_type=RENDER_POSE_OUT_TYPE_RGB) {
  int nLimbs = limbSeq.size() / 2;
  int nchannels = 3;
  if (out_type == RENDER_POSE_OUT_TYPE_RGB) {
    nchannels = 3;
  } else if (out_type == RENDER_POSE_OUT_TYPE_SPLITCHANNEL) {
    nchannels = nLimbs;
  } else {
    cerr << "render_pose: Unknown output type." << endl;
  }
  Mat output(out_ht, out_wd, CV_32FC(nchannels), 0.0);
  vector<Mat> output_channels;
  if (nchannels != 3) {
    split(output, output_channels);
  }
  // assert(limbSeq.size() / 2 == joint_color.size() / 3);
  for (int body_id = 0; body_id < poses.size(); body_id++) {
    for (int i = 0; i < nLimbs; i++) {
      float scal_ht = out_ht * 1.0 / max_ht;
      float scal_wd = out_wd * 1.0 / max_wd;
      tuple<float, float, float> pt1 = poses[body_id][limbSeq[2*i]-1];
      tuple<float, float, float> pt2 = poses[body_id][limbSeq[2*i+1]-1];
      float pt1_conf = get<2>(pt1);
      float pt2_conf = get<2>(pt2);
      if (pt1_conf < 0.1 || pt2_conf < 0.1) {
        continue;
      }
      Mat render_img;
      Scalar color;
      if (nchannels == 3) {
        render_img = output;
        color = CV_RGB(joint_color[i*3], joint_color[i*3+1], joint_color[i*3+2]);
      } else {
        render_img = output_channels[i];
        color = Scalar(1);
      }
      line(
          render_img,
          Point(get<0>(pt1) * scal_wd, get<1>(pt1) * scal_ht),
          Point(get<0>(pt2) * scal_wd, get<1>(pt2) * scal_ht),
          color, marker_wd);
    }
  }
  if (nchannels != 3) {
    merge(output_channels, output);
  }
  return output;
}


vector<vector<tuple<float,float,float>>> read_pose_xml(string xml_str, int &pose_dim) {
  vector<vector<tuple<float,float,float>>> poses;
  if (xml_str.size() > 0) {
    stringstream ss(xml_str);
    pt::ptree root;
    pt::read_json(ss, root);
    for (pt::ptree::value_type &body : root.get_child("bodies")) {
      vector<float> elts;
      for (pt::ptree::value_type &joints : body.second.get_child("joints")) {
        elts.push_back((float) stof(joints.second.data()));
      }
      pose_dim = elts.size() / 3;  // x,y,score format
      if (pose_dim * 3 != elts.size()) {
        cerr << "Invalid number of numbers in pose dim ("
          << pose_dim * 3 << " vs " << elts.size() << endl;
        poses.clear();
        break;
      }
      vector<tuple<float,float,float>> pose;
      for (int i = 0; i < pose_dim; i++) {
        pose.push_back(make_tuple(elts[i*3], elts[i*3+1], elts[i*3+2]));
      }
      poses.push_back(pose);
    }
  } else {
    cerr << "json_to_pose: Empty string passed in." << endl;
  }
  return poses;
}
