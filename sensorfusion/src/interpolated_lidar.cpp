#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"


#include <limits>
#include <iostream>
#include <cmath>
#include "pcl/console/parse.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/io/pcd_io.h"
#include "pcl/sample_consensus/ransac.h"
#include "pcl/sample_consensus/sac_model_plane.h"
#include "pcl/sample_consensus/sac_model_sphere.h"
#include "pcl/point_types.h"
#include "pcl/range_image/range_image.h"
#include "pcl/range_image/range_image_spherical.h"
#include "pcl/filters/filter.h"
#include "pcl/filters/voxel_grid.h"
#include "pcl/filters/statistical_outlier_removal.h"
#include "pcl/point_cloud.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include "pcl_conversions/pcl_conversions.h"
#include "image_transport/image_transport.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "Eigen/Dense"
#include "armadillo"

#include <chrono>


typedef std::chrono::high_resolution_clock Clock;

using namespace Eigen;
using namespace sensor_msgs;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub;
rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr imgD_pub;

float maxlen =100.0;     //maxima distancia del lidar
float minlen = 0.01;     //minima distancia del lidar
float max_FOV = 3.0;     // en radianes angulo maximo de vista de la camara
float min_FOV = 0.4;     // en radianes angulo minimo de vista de la camara

/// parametros para convertir nube de puntos en imagen
float angular_resolution_x =0.5f;
float angular_resolution_y = 2.1f;
float max_angle_width= 360.0f;
float max_angle_height = 180.0f;

float interpol_value = 20.0;
float ang_x_lidar = 0.6*M_PI/180.0; 
double max_var = 50.0; 
bool f_pc = true; 


// topics a suscribirse del nodo
std::string pcTopic = "/velodyne_points";


// range image parametros
boost::shared_ptr<pcl::RangeImageSpherical> rangeImage;
pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::LASER_FRAME;

///////////////////////////////////////callback
void callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg_pointCloud)
{
  ////// filter point cloud 
  if (!msg_pointCloud) return;

  PointCloud::Ptr cloud_in (new PointCloud);
  PointCloud::Ptr cloud_out (new PointCloud);

  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*msg_pointCloud, *cloud_in, indices);

  float max_z = 0, min_z = std::numeric_limits<float>::infinity();
  float max_dis = 0, min_dis = std::numeric_limits<float>::infinity();

  for (int i = 0; i < (int) cloud_in->points.size(); i++) 
  {
    double distance = sqrt(cloud_in->points[i].x * cloud_in->points[i].x + cloud_in->points[i].y * cloud_in->points[i].y);
    if (distance < minlen || distance > maxlen)
        continue;

    cloud_out->push_back(cloud_in->points[i]);
    if (cloud_in->points[i].z > max_z)
        max_z = cloud_in->points[i].z;
    if (cloud_in->points[i].z < min_z)
        min_z = cloud_in->points[i].z;
    if (distance > max_dis)
        max_dis = distance;
    if (distance < min_dis)
        min_dis = distance;
  }


  //                                  point cloud to image 

  //============================================================================================================
  //============================================================================================================

  // range image    

  Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, 0.0f);

  rangeImage->pcl::RangeImage::createFromPointCloud(*cloud_out, pcl::deg2rad(angular_resolution_x), pcl::deg2rad(angular_resolution_y),
                                       pcl::deg2rad(max_angle_width), pcl::deg2rad(max_angle_height),
                                       sensorPose, coordinate_frame, 0.0f, 0.0f, 0);

  int cols_img = rangeImage->width;
  int rows_img = rangeImage->height;

  arma::mat Z;  // Interpolation of the image
  arma::mat Zz; // Interpolation of image heights

  Z.zeros(rows_img, cols_img); // Range
  Zz.zeros(rows_img, cols_img); // Height

  Eigen::MatrixXf ZZei(rows_img, cols_img);

  float max_depth = 0.0;
  float min_depth = -999.0;

  for (int i = 0; i < cols_img; ++i) 
  {
    for (int j = 0; j < rows_img; ++j) 
    {
        float r = rangeImage->getPoint(i, j).range;
        float zz = rangeImage->getPoint(i, j).z;

        Eigen::Vector3f tmp_point;
        rangeImage->calculate3DPoint(float(i), float(j), r, tmp_point);

        if (std::isinf(r) || r < minlen || r > maxlen || std::isnan(zz)) {
            continue;
        }

        Z.at(j, i) = r;
        Zz.at(j, i) = zz;
        ZZei(j, i) = zz;

        if (r > max_depth) {
            max_depth = r;
        }
        if (r < min_depth) {
            min_depth = r;
        }
    }
  }


  ////////////////////////////////////////////// interpolation
  //============================================================================================================
  
  arma::vec X = arma::regspace(1, Z.n_cols);  // X = horizontal spacing
  arma::vec Y = arma::regspace(1, Z.n_rows);  // Y = vertical spacing

  arma::vec XI = arma::regspace(X.min(), 1.0, X.max()); // magnify by approx 2
  arma::vec YI = arma::regspace(Y.min(), 1.0 / interpol_value, Y.max()); // Y solo para poner el suelo

  arma::mat ZI;
  arma::mat ZzI;

  arma::interp2(ZI, X, Y, Z, XI, YI, "lineal");
  arma::interp2(ZzI, X, Y, Zz, XI, YI, "lineal");

  arma::mat Zout = ZI;

  for (uint i = 0; i < ZI.n_rows; ++i) {
    for (uint j = 0; j < ZI.n_cols; ++j) {
        if (ZI(i, j) == 0) {
            if (i + interpol_value < ZI.n_rows) {
                for (int k = 1; k <= interpol_value; k+=1) {
                    Zout(i + k, j) = 0;
                }
            }
            if (i > interpol_value) {
                for (int k = 1; k <= interpol_value; k+=1) {
                    Zout(i - k, j) = 0;
                }
            }
        }
    }
  }

  ZI = Zout;

  if (f_pc){    
    //////////////////filtrado de elementos interpolados con el fondo
    
    /// filtrado por varianza
    for (uint i = 0; i < ((ZI.n_rows - 1) / interpol_value); i+=1) {
      for (uint j = 0; j < ZI.n_cols - 5; j+=1) {
          double promedio = 0;
          double varianza = 0;
          for (uint k = 0; k < interpol_value; k+=1) {
              promedio = promedio + ZI((i * interpol_value) + k, j);
          }
          promedio = promedio / interpol_value;

          for (uint l = 0; l < interpol_value; l++) {
              varianza = varianza + pow((ZI((i * interpol_value) + l, j) - promedio), 2.0);
          }
          varianza = sqrt(varianza / interpol_value);

          if (varianza > max_var) {
              for (uint m = 0; m < interpol_value; m++) {
                  Zout((i * interpol_value) + m, j) = 0;
              }
          }
      }
    }
  }
  ZI = Zout;


  // reconstruccion de imagen a nube 3D
  //============================================================================================================
  
  PointCloud::Ptr point_cloud (new PointCloud);
  PointCloud::Ptr P_out (new PointCloud);  
  point_cloud->width = ZI.n_cols;
  point_cloud->height = ZI.n_rows;
  point_cloud->is_dense = false;
  point_cloud->points.resize(point_cloud->width * point_cloud->height);

  int num_pc = 0; // numero de elementos en pointcloud
  for (uint i = 0; i < ZI.n_rows - interpol_value; i += 1) {
    for (uint j = 0; j < ZI.n_cols; j += 1) {
        float ang = M_PI - ((2.0 * M_PI * j) / (ZI.n_cols));

        if (!(Zout(i, j) == 0)) {
            float pc_modulo = Zout(i, j);
            float pc_x = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * cos(ang);
            float pc_y = sqrt(pow(pc_modulo, 2) - pow(ZzI(i, j), 2)) * sin(ang);

            Eigen::MatriXf Lidar_matrix(3, 3); // Replace MatrixXf with Matrix3f
            Eigen::VectorXf result(3, 1);
            Lidar_matrix << cos(ang_x_lidar), 0, sin(ang_x_lidar),
                            0, 1, 0,
                            -sin(ang_x_lidar), 0, cos(ang_x_lidar);

            result << pc_x,
                      pc_y,
                      ZzI(i, j);

            result = Lidar_matrix * result;

            point_cloud->points[num_pc].x = result(0);
            point_cloud->points[num_pc].y = result(1);
            point_cloud->points[num_pc].z = result(2);
            P_out->push_back(point_cloud->points[num_pc]);

            num_pc++;
        }
    }
  }

  int size_inter_Lidar = (int) P_out->points.size(); 
  uint px_data = 0; uint py_data = 0;
  pcl::PointXYZ point;

  P_out->is_dense = true;
  P_out->width = (int) P_out->points.size();
  P_out->height = 1;
  P_out->header.frame_id = "velodyne";
  pc_pub->publish(P_out);

 // cv::Mat interdephtImage =  cv::Mat::zeros(ZI.n_rows, ZI.n_cols*2, cv_bridge::getCvType("mono16"));
  cv::Mat interdephtImage =  cv::Mat::zeros(ZI.n_rows, ZI.n_cols, cv_bridge::getCvType("mono16"));

  for (int i=0; i< ZI.n_cols; ++i)
      for (int j=0; j<ZI.n_rows ; ++j)

      {
        interdephtImage.at<ushort>(j, i) = 1-(pow(2,16)/ (maxlen - minlen))*( ZI(j,i)-minlen);   
        //interdephtImage.at<ushort>(j, i+ZI.n_cols) = (ZzI(j,i)/20.0)* pow(2,16);     
      }

  auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", interdephtImage).toImageMsg();
  imgD_pub->publish(image_msg);
}


int main(int argc, char** argv)
{
  // Initialize ROS2
  rclcpp::init(argc, argv);

  // Create a node
  auto node = rclcpp::Node::make_shared("InterpolatedPointCloud");
 
  // Retrieve parameters
  node->get_parameter("/maxlen", maxlen);
  node->get_parameter("/minlen", minlen);
  node->get_parameter("/pcTopic", pcTopic);
  node->get_parameter("/x_resolution", angular_resolution_x);
  node->get_parameter("/y_interpolation", interpol_value);
  node->get_parameter("/ang_Y_resolution", angular_resolution_y);
  node->get_parameter("/ang_ground", ang_x_lidar);
  node->get_parameter("/max_var", max_var);
  node->get_parameter("/filter_output_pc", f_pc);

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(pcTopic, 10, callback);
  rangeImage = boost::shared_ptr<pcl::RangeImageSpherical>(new pcl::RangeImageSpherical);
  
  pc_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("/pc_interpoled", 10);
  imgD_pub = node->create_publisher<sensor_msgs::msg::Image>("/pc2imageInterpol", 10);
  rclcpp::spin(node);
}