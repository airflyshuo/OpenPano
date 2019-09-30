// File: main.cc
// Date: Wed Jun 17 20:29:58 2015 +0800
// Author: Yuxin Wu

#define _USE_MATH_DEFINES
#include <cmath>

#include "feature/extrema.hh"
#include "feature/matcher.hh"
#include "feature/orientation.hh"
#include "lib/mat.h"
#include "lib/config.hh"
#include "lib/geometry.hh"
#include "lib/imgproc.hh"
#include "lib/planedrawer.hh"
#include "lib/polygon.hh"
#include "lib/timer.hh"
#include "stitch/cylstitcher.hh"
#include "stitch/match_info.hh"
#include "stitch/stitcher.hh"
#include "stitch/transform_estimate.hh"
#include "stitch/warp.hh"
#include "common/common.hh"
#include <ctime>
#include <cassert>

#ifdef DISABLE_JPEG
#define IMGFILE(x) #x ".png"
#else
#define IMGFILE(x) #x ".jpg"
#endif

using namespace std;
using namespace pano;
using namespace config;

bool TEMPDEBUG = false;

const int LABEL_LEN = 7;

void test_extrema(const char* fname, int mode) {
	auto mat = read_img(fname);

	ScaleSpace ss(mat, NUM_OCTAVE, NUM_SCALE);
	DOGSpace dog(ss);
	ExtremaDetector ex(dog);

	PlaneDrawer pld(mat);
	if (mode == 0) {
		auto extrema = ex.get_raw_extrema();
		PP(extrema.size());
		for (auto &i : extrema)
			pld.cross(i, LABEL_LEN / 2);
	} else if (mode == 1) {
		auto extrema = ex.get_extrema();
		cout << extrema.size() << endl;
		for (auto &i : extrema) {
			Coor c{(int)(i.real_coor.x * mat.width()), (int)(i.real_coor.y * mat.height())};
			pld.cross(c, LABEL_LEN / 2);
		}
	}
	write_rgb(IMGFILE(extrema), mat);
}

void test_orientation(const char* fname) {
	auto mat = read_img(fname);
	ScaleSpace ss(mat, NUM_OCTAVE, NUM_SCALE);
	DOGSpace dog(ss);
	ExtremaDetector ex(dog);
	auto extrema = ex.get_extrema();
	OrientationAssign ort(dog, ss, extrema);
	auto oriented_keypoint = ort.work();

	PlaneDrawer pld(mat);
	pld.set_rand_color();

	cout << "FeaturePoint size: " << oriented_keypoint.size() << endl;
	for (auto &i : oriented_keypoint)
		pld.arrow(Coor(i.real_coor.x * mat.width(), i.real_coor.y * mat.height()), i.dir, LABEL_LEN);
	write_rgb(IMGFILE(orientation), mat);
}

// draw feature and their match
void test_match(const char* f1, const char* f2) {
	list<Mat32f> imagelist;
	Mat32f pic1 = read_img(f1);
	Mat32f pic2 = read_img(f2);
	imagelist.push_back(pic1);
	imagelist.push_back(pic2);

	unique_ptr<FeatureDetector> detector;
	detector.reset(new SIFTDetector);
	vector<Descriptor> feat1 = detector->detect_feature(pic1),
										 feat2 = detector->detect_feature(pic2);
	print_debug("Feature: %lu, %lu\n", feat1.size(), feat2.size());

	Mat32f concatenated = hconcat(imagelist);
	PlaneDrawer pld(concatenated);

	FeatureMatcher match(feat1, feat2);
	auto ret = match.match();
	print_debug("Match size: %d\n", ret.size());
	for (auto &x : ret.data) {
		pld.set_rand_color();
		Vec2D coor1 = feat1[x.first].coor,
					coor2 = feat2[x.second].coor;
		Coor icoor1 = Coor(coor1.x + pic1.width()/2, coor1.y + pic1.height()/2);
		Coor icoor2 = Coor(coor2.x + pic2.width()/2 + pic1.width(), coor2.y + pic2.height()/2);
		pld.circle(icoor1, LABEL_LEN);
		pld.circle(icoor2, LABEL_LEN);
		pld.line(icoor1, icoor2);
	}
	write_rgb(IMGFILE(match), concatenated);
}

// draw inliers of the estimated homography
void test_inlier(const char* f1, const char* f2) {
	list<Mat32f> imagelist;
	Mat32f pic1 = read_img(f1);
	Mat32f pic2 = read_img(f2);
	imagelist.push_back(pic1);
	imagelist.push_back(pic2);

	unique_ptr<FeatureDetector> detector;
	detector.reset(new SIFTDetector);
	vector<Descriptor> feat1 = detector->detect_feature(pic1),
										 feat2 = detector->detect_feature(pic2);
	vector<Vec2D> kp1; for (auto& d : feat1) kp1.emplace_back(d.coor);
	vector<Vec2D> kp2; for (auto& d : feat2) kp2.emplace_back(d.coor);
	print_debug("Feature: %lu, %lu\n", feat1.size(), feat2.size());

	Mat32f concatenated = hconcat(imagelist);
	PlaneDrawer pld(concatenated);
	FeatureMatcher match(feat1, feat2);
	auto ret = match.match();
	print_debug("Match size: %d\n", ret.size());

	TransformEstimation est(ret, kp1, kp2,
			{pic1.width(), pic1.height()}, {pic2.width(), pic2.height()});
	MatchInfo info;
	est.get_transform(&info);
	print_debug("Inlier size: %lu, conf=%lf\n", info.match.size(), info.confidence);
	if (info.match.size() == 0)
		return;

	for (auto &x : info.match) {
		pld.set_rand_color();
		Vec2D coor1 = x.first,
					coor2 = x.second;
		Coor icoor1 = Coor(coor1.x + pic1.width()/2, coor1.y + pic1.height()/2);
		Coor icoor2 = Coor(coor2.x + pic2.width()/2, coor2.y + pic2.height()/2);
		pld.circle(icoor1, LABEL_LEN);
		pld.circle(icoor2 + Coor(pic1.width(), 0), LABEL_LEN);
		pld.line(icoor1, icoor2 + Coor(pic1.width(), 0));
	}
	pld.set_color(Color(0,0,0));
	Vec2D offset1(pic1.width()/2, pic1.height()/2);
	Vec2D offset2(pic2.width()/2 + pic1.width(), pic2.height()/2);

	// draw convex hull of inliers
	/*
	 *vector<Vec2D> pts1, pts2;
	 *for (auto& x : info.match) {
	 *  pts1.emplace_back(x.first + offset1);
	 *  pts2.emplace_back(x.second + offset2, 0));
	 *}
	 *auto hull = convex_hull(pts1);
	 *pld.polygon(hull);
	 *hull = convex_hull(pts2);
	 *pld.polygon(hull);
	 */

	// draw warped four edges
	Shape2D shape2{pic2.width(), pic2.height()}, shape1{pic1.width(), pic1.height()};

	// draw overlapping region
	Matrix homo(3,3);
	REP(i, 9) homo.ptr()[i] = info.homo[i];
	Homography inv = info.homo.inverse();
	auto p = overlap_region(shape1, shape2, homo, inv);
	PA(p);
	for (auto& v: p) v += offset1;
	pld.polygon(p);

	Matrix invM(3, 3);
	REP(i, 9) invM.ptr()[i] = inv[i];
	p = overlap_region(shape2, shape1, invM, info.homo);
	PA(p);
	for (auto& v: p) v += offset2;
	pld.polygon(p);

	write_rgb(IMGFILE(inlier), concatenated);
}

void test_warp(int argc, char* argv[]) {
	CylinderWarper warp(1);
	REPL(i, 2, argc) {
		Mat32f mat = read_img(argv[i]);
		warp.warp(mat);
		write_rgb(("warp" + to_string(i) + ".jpg").c_str(), mat);
	}
}


void work(int argc, char* argv[]) {
/*
 *  vector<Mat32f> imgs(argc - 1);
 *  {
 *    GuardedTimer tm("Read images");
 *#pragma omp parallel for schedule(dynamic)
 *    REPL(i, 1, argc)
 *      imgs[i-1] = read_img(argv[i]);
 *  }
 */
	vector<string> imgs;
	REPL(i, 1, argc) imgs.emplace_back(argv[i]);
	Mat32f res;
	if (CYLINDER) {
		CylinderStitcher p(move(imgs));
		res = p.build();
	} else {
		Stitcher p(move(imgs));
		res = p.build();
	}

	if (CROP) {
		int oldw = res.width(), oldh = res.height();
		res = crop(res);
		print_debug("Crop from %dx%d to %dx%d\n", oldw, oldh, res.width(), res.height());
	}
	{
		GuardedTimer tm("Writing image");
		write_rgb(IMGFILE(out), res);
	}
}

void init_config() {
#define CFG(x) \
	x = Config.get(#x)
	const char* config_file = "config.cfg";
	ConfigParser Config(config_file);
	CFG(CYLINDER);
	CFG(TRANS);
	CFG(ESTIMATE_CAMERA);
	if (int(CYLINDER) + int(TRANS) + int(ESTIMATE_CAMERA) >= 2)
		error_exit("You set two many modes...\n");
	if (CYLINDER)
		print_debug("Run with cylinder mode.\n");
	else if (TRANS)
		print_debug("Run with translation mode.\n");
	else if (ESTIMATE_CAMERA)
		print_debug("Run with camera estimation mode.\n");
	else
		print_debug("Run with naive mode.\n");

	CFG(ORDERED_INPUT);
	if (!ORDERED_INPUT && !ESTIMATE_CAMERA)
		error_exit("Require ORDERED_INPUT under this mode!\n");

	CFG(CROP);
	CFG(STRAIGHTEN);
	CFG(FOCAL_LENGTH);
	CFG(MAX_OUTPUT_SIZE);
	CFG(LAZY_READ);	// TODO in cyl mode

	CFG(SIFT_WORKING_SIZE);
	CFG(NUM_OCTAVE);
	CFG(NUM_SCALE);
	CFG(SCALE_FACTOR);
	CFG(GAUSS_SIGMA);
	CFG(GAUSS_WINDOW_FACTOR);
	CFG(JUDGE_EXTREMA_DIFF_THRES);
	CFG(CONTRAST_THRES);
	CFG(PRE_COLOR_THRES);
	CFG(EDGE_RATIO);
	CFG(CALC_OFFSET_DEPTH);
	CFG(OFFSET_THRES);
	CFG(ORI_RADIUS);
	CFG(ORI_HIST_SMOOTH_COUNT);
	CFG(DESC_HIST_SCALE_FACTOR);
	CFG(DESC_INT_FACTOR);
	CFG(MATCH_REJECT_NEXT_RATIO);
	CFG(RANSAC_ITERATIONS);
	CFG(RANSAC_INLIER_THRES);
	CFG(INLIER_IN_MATCH_RATIO);
	CFG(INLIER_IN_POINTS_RATIO);
	CFG(SLOPE_PLAIN);
	CFG(LM_LAMBDA);
	CFG(MULTIPASS_BA);
	CFG(MULTIBAND);
#undef CFG
}

bool GetIntersection(double u, double v,
  double &x, double &y, double &z)
{
  double Nx    = 0.0;
  double Ny    = 0.0;
  double Nz    = 1.0;
  double dir_x = u - Nx;
  double dir_y = v - Ny;
  double dir_z = -1.0 - Nz;
 
  double a = (dir_x * dir_x) + (dir_y * dir_y) + (dir_z * dir_z);
  double b = (dir_x * Nx) + (dir_y * Ny) + (dir_z * Nz);
 
  b *= 2;
  double d = b*b;
  double q = -0.5 * (b - std::sqrt(d));
 
  double t = q / a;
 
  x = (dir_x * t) + Nx;
  y = (dir_y * t) + Ny;
  z = (dir_z * t) + Nz;
  return true;
}

const int PLANET_OUT_W = 1920, PLANET_OUT_H = 1080;
string planet_out_dir = "./planet_my_pano";
/**
 * actual_w: The width of the actual area of the image
 * fov: field of view
 * pitch: angle of x-axis rotation
 * roll:  angle of y-axis rotation
 * yaw:   angle of z-axis rotation
 */
void planet(const Mat32f panoImg, int actual_w, const double fov, const double pitch, const double roll, const double yaw) {
	int w = panoImg.width(), h = panoImg.height();
	const int center_w = PLANET_OUT_W / 2, center_h = PLANET_OUT_H / 2;
	update_min(actual_w, PLANET_OUT_W);

	Mat32f ret(PLANET_OUT_H, PLANET_OUT_W, 3);
	fill(ret, Color::BLACK);

	REPL(i, center_w - actual_w/2, center_w + actual_w/2) REP(j, PLANET_OUT_H) {
		double sphereX = (i - center_w) * 4.0 * std::tan((fov/2) * M_PI / 180.0) / PLANET_OUT_H;
		double sphereY = (j - center_h) * 4.0 * std::tan((fov/2) * M_PI / 180.0) / PLANET_OUT_H;
		double Qx, Qy, Qz;

		if (GetIntersection(sphereX, sphereY, Qx, Qy, Qz))
		{
			if(pitch != 0)
			{
				double theta_x = pitch * M_PI / 180.0;
				double rot_y = Qy * cos(theta_x) + Qz * sin(theta_x);
				double rot_z = Qy * sin(theta_x) * -1 + Qz * cos(theta_x);
				Qy = rot_y; Qz = rot_z;
			}
			if(roll != 0) 
			{
				double theta_y = roll * M_PI / 180.0;
				double rot_x = Qx * cos(theta_y) - Qz * sin(theta_y);
				double rot_z = Qx * sin(theta_y) + Qz * cos(theta_y);
				Qx = rot_x; Qz = rot_z;
			}
			if(yaw != 0)
			{
				double theta_z = yaw * M_PI / 180.0;
				double rot_x = Qx * cos(theta_z) + Qy * sin(theta_z);
				double rot_y = Qx * sin(theta_z) * -1 + Qy * cos(theta_z);
				Qx = rot_x; Qy = rot_y;
			}
			
			double theta = std::acos(Qz);
			double phi   = std::atan2(Qy, Qx) + M_PI;
			theta        = theta * M_1_PI;
			phi          = phi   * (0.5 * M_1_PI);
			double Sx    = min(w - 2.0, w * phi);
			double Sy    = min(h - 2.0, h * theta);

			Color c = interpolate(panoImg, Sy, Sx);
			float* p = ret.ptr(j, i);
			c.write_to(p);
		}
	}
	
	write_rgb(ssprintf("%s/fov%.2f_pitch%.2f_roll%.2f_yaw%.2f.jpg", planet_out_dir.data(), fov, pitch, roll, yaw), ret);
}

void planet(const char* fname, const double fov, const double pitch, const double roll, const double yaw) {
	string srcFile = fname;
	if(srcFile.find("dji") != string::npos || srcFile.find("DJI") != string::npos)
		planet_out_dir = "./planet_dji_pano";

	planet(read_img(fname), PLANET_OUT_W, fov, pitch, roll, yaw);
}

void planet(const Mat32f panoImg, const double fov, const double pitch, const double roll, const double yaw) {
	planet(panoImg, PLANET_OUT_W, fov, pitch, roll, yaw);
}

/**
 * Generate frames of the animation of planet
 */
void planet_ani(const char* fname, const int part) {
	Mat32f panoImg = read_img(fname);
	string srcFile = fname;
	if(srcFile.find("dji") != string::npos || srcFile.find("DJI") != string::npos)
		planet_out_dir = "./planet_dji_pano";

	//230-299 frames
	double fov = 110, pitch = 0.0, roll = 0.0, yaw = 135.0;
	if(part >= 1)
	{
		for(int i = 0; i < 70; i++)
		{
			// if(part == 1)
				planet(panoImg, PLANET_OUT_H + i*12, fov, pitch, roll, yaw);
			fov -= 0.1;
			pitch += 0.01;
			roll -= 0.01;
			yaw += 0.2;
		}
	}
	//160-229 frames
	//start from fov103.00_pitch0.70_roll-0.70_yaw149.00
	if(part >= 2)
	{
		double s_pitch = pitch;
		double s_roll = roll;
		for(int i = 0; i < 70; i++)
		{
			double dx = sin(i * M_PI/70);
			double dy = cos(i * M_PI/70);
			double pitch_offset = (1 - dy) * (11.2 - 0.7);
			double roll_offset = dx * (11.2 - 0.7);
			pitch = s_pitch + pitch_offset;
			roll = s_roll - roll_offset;
			// if(part == 2)
				planet(panoImg, fov, pitch, roll, yaw);
			fov -= 0.8;
			yaw += 1.74;
		}
	}
	//150-159 frames
	if(part == 3)
	{
		double roll_offset = roll/10.0;
		for(int i = 0; i < 10; i++)
		{
			roll -= roll_offset;
			planet(panoImg, fov, pitch, roll, 270.0);
			fov -= 1.5;
		}
	}
}

int main(int argc, char* argv[]) {
	if (argc <= 2)
		error_exit("Need at least two images to stitch.\n");
	TotalTimerGlobalGuard _g;
	srand(time(NULL));
	init_config();
	string command = argv[1];
	if (command == "raw_extrema")
		test_extrema(argv[2], 0);
	else if (command == "keypoint")
		test_extrema(argv[2], 1);
	else if (command == "orientation")
		test_orientation(argv[2]);
	else if (command == "match")
		test_match(argv[2], argv[3]);
	else if (command == "inlier")
		test_inlier(argv[2], argv[3]);
	else if (command == "warp")
		test_warp(argc, argv);
	else if (command == "planet")
		planet(argv[2], strtod(argv[3], NULL), strtod(argv[4], NULL), strtod(argv[5], NULL), strtod(argv[6], NULL));
	else if (command == "planet_ani")
		planet_ani(argv[2], strtol(argv[3], NULL, 10));
	else
		// the real routine
		work(argc, argv);
}
