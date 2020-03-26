#include <bathy_mapper/bathy_mapper_nodelet.h>

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(bathy_mapper::BathyMapperNodelet, nodelet::Nodelet)

namespace bathy_mapper
{
void BathyMapperNodelet::onInit()
{
	ros::NodeHandle& nh = getNodeHandle();
	ros::NodeHandle& nh_priv = getPrivateNodeHandle();

	if (nh_priv.param("multithreaded", false))
	{
		nh = getMTNodeHandle();
		nh_priv = getMTPrivateNodeHandle();
	}

    server_ = std::make_shared<BathyMapper>(nh, nh_priv);
}
}  // namespace bathy_mapper
