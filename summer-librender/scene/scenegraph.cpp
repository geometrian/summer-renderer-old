#include "scenegraph.hpp"

#include "object.hpp"


namespace Summer { namespace Scene {


void Node::_register_for_build(
	OptiX::AccelerationStructure::BuilderInstances* builder,
	Mat4x4f const& transform_wld_to_outside
) const {
	Mat4x4f transform_wld_to_inside = transform_wld_to_outside;
	if      (transform.type==Transform::TYPE::NONE  );
	else if (transform.type==Transform::TYPE::MATRIX) {
		transform_wld_to_inside = transform_wld_to_inside * transform.matrix;
	} else {
		if ((transform.type&Transform::TYPE::MSK_TRANSLATE)!=0) {
			transform_wld_to_inside = glm::translate( transform_wld_to_inside, transform.translate );
		}
		if ((transform.type&Transform::TYPE::MSK_ROTATE)!=0) {
			glm::quat quaternion( transform.rotate.w, transform.rotate.x,transform.rotate.y,transform.rotate.z );
			Mat4x4f rotation = static_cast<Mat4x4f>(quaternion);
			transform_wld_to_inside = transform_wld_to_inside * rotation;
		}
		if ((transform.type&Transform::TYPE::MSK_SCALE)!=0) {
			transform_wld_to_inside = glm::scale( transform_wld_to_inside, transform.scale );
		}
	}
	Mat3x4f transform_wld_to_inside_3x4 = Mat3x4f(transform_wld_to_inside);

	for (Node const* child : children) {
		child->_register_for_build(builder,transform_wld_to_inside_3x4);
	}
	for (Object const* object : objects) {
		builder->add_instance(object->accel->handle,transform_wld_to_inside);
	}
}


Scene::Scene() {
	accel = nullptr;
}
Scene::~Scene() {
	 delete accel;
}

void Scene::upload(OptiX::Context const* context_optix) {
	OptiX::AccelerationStructure::BuilderInstances builder;
	for (Node const* node : root_nodes) node->_register_for_build(&builder,Mat4x4f(
		1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1
	));
	builder.finish();

	accel = new OptiX::AccelerationStructure(context_optix,builder);
}


SceneGraph::~SceneGraph() {
	for (Camera const* camera : cameras) delete camera;

	for (Scene const* scene : scenes) delete scene;

	for (Node const* node : nodes) delete node;

	for (Object const* object : objects) delete object;

	for (DataBlock::AccessorBase const* datablock_accessor : datablock_accessors) delete datablock_accessor;
	for (DataBlock::View         const* datablock_view     : datablock_views    ) delete datablock_view;
	for (DataBlock               const* datablock          : datablocks         ) delete datablock;
}

void SceneGraph::upload(OptiX::Context const* context_optix) {
	//Upload data buffers
	for (DataBlock* datablock : datablocks) datablock->upload();

	//Build and upload objects' acceleration structures
	for (Object* object : objects) object->upload(context_optix);

	//Build and upload the instance acceleration structure for each scene
	for (Scene* scene : scenes) scene->upload(context_optix);
}


}}
