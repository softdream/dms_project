#ifndef __PREDICT_H_
#define __PREDICT_H_

#include <type_traits>
#include <iostream>

template<typename T>
struct is_double_or_float
{
	static const bool value = false;
};

template<>
struct is_double_or_float<float>
{
	static const bool value = true;
};

template<>
struct is_double_or_float<double>
{
	static const bool value = true;
};


class Predict
{
public:
	Predict()
	{

	}

	explicit Predict(const double delta_t_) : delta_t(delta_t_)
	{

	}

	/** 描述：根据常量速度模型来预测 delta_t 时间以后的目标位置
	*	参数：velocity_x ： 输入 x 轴方向的速度值；
	*   参数：velocity_y ： 输入 y 轴方向的速度值；
	*   参数：pose_x_old ： 输入当前时刻的 x 轴方向的空间坐标值；
	*   参数：pose_y_old ： 输入当前时刻的 y 轴方向的空间坐标值；
	*   参数：pose_x_new ： 输出下一时刻（即delta_t事件以后）的 x 轴方向的空间坐标值；
	*   参数：pose_y_new ： 输出下一时刻（即delta_t事件以后）的 y 轴方向的空间坐标值；
	*   返回值 ： bool;
	*/

	template<typename T, typename = typename std::enable_if<is_double_or_float<T>::value>::type>
	inline bool operator() (const T &velocity_x, const T &velocity_y, 
					 const T &pose_x_old, const T &pose_y_old,
					 T &pose_x_new, T&pose_y_new
					 ) const
	{
		pose_x_new = pose_x_old + velocity_x * delta_t;
		pose_y_new = pose_y_old + velocity_y * delta_t;

		return true;
	}

private:
	double delta_t = 0.04; // 预测时间间隔，默认 40 毫秒
};

#endif
