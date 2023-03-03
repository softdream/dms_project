#pragma once

//#include <opencv2/core.hpp>
#include "kalman_filter.h"

typedef struct Point_
{
	Point_()
	{

	}

	Point_(const float x_, const float y_) : x(x_), y(y_)
	{

	}

	float x;
	float y;
}Point;

typedef struct Rect_
{
	Rect_()
	{

	}

	Rect_(const int x_, const int y_, const int width_, const int height_) : x(x_), y(y_), width(width_), height(height_)
	{

	}

	const Point tl() const // ?????
	{
		return Point(x, y);
	}

	const Point br() const // ?????
	{
		return Point(x + width, y + height);
	}

	const float area() const
	{
		return width * height;
	}

	int x; /* ???¦Å???????x-???? */

	int y; /* ???¦Å???????y-????*/

	int width; /* ?? */

	int height; /* ?? */
}Rect;

class Track {
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const Rect& bbox);
    void Predict();
    void Update(const Rect& bbox);
    Rect GetStateAsBbox() const;
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

private:
    Eigen::VectorXd ConvertBboxToObservation(const Rect& bbox) const;
    Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;

    KalmanFilter kf_;
};
