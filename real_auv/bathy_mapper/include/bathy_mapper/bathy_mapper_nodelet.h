#ifndef BATHY_MAPPER_NODELET_H
#define BATHY_MAPPER_NODELET_H

#include <nodelet/nodelet.h>

#include <bathy_mapper/bathy_mapper.h>

#include <memory>

namespace bathy_mapper
{
class BathyMapperNodelet : public nodelet::Nodelet
{
private:
    std::shared_ptr<BathyMapper> server_;

public:
	void onInit() override;
};
}  // namespace bathy_mapper

#endif  // BATHY_MAPPER_NODELET_H
