#include "gltf.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "object.hpp"
#include "scenegraph.hpp"


namespace Summer { namespace Scene {


SceneGraph* load_new_gltf(std::string const& path) {
	//Load file
	tinygltf::TinyGLTF gltf_ctx;
	tinygltf::Model asset;
	std::string err, warn;
	bool ret = false;
	if (IB::Str::endswith(path,"glb")) {
		ret = gltf_ctx.LoadBinaryFromFile( &asset, &err,&warn, path );
	} else {
		ret = gltf_ctx.LoadASCIIFromFile ( &asset, &err,&warn, path );
	}
	assert_term(ret,"Could not load GTLF file!");

	//Convert to result
	SceneGraph* result = new SceneGraph;
	//	Datablocks (GLTF "buffer"s)
	for (tinygltf::Buffer const& iter : asset.buffers) {
		result->datablocks.emplace_back(new DataBlock(
			iter.data
		));
	}
	//	Datablock views (GLTF "buffer view"s)
	for (tinygltf::BufferView const& iter : asset.bufferViews) {
		assert_term(iter.buffer>=0,"Implementation error!");
		size_t datablock_index = static_cast<size_t>(iter.buffer);
		assert_term(datablock_index<result->datablocks.size(),"Invalid datablock index!");

		result->datablock_views.emplace_back(new DataBlock::View(
			result->datablocks[datablock_index],
			iter.byteOffset, iter.byteStride, iter.byteLength
		));
	}
	//	Datablock accessors (GLTF "buffer accessor"s)
	for (tinygltf::Accessor const& iter : asset.accessors) {
		assert_term(iter.bufferView>=0,"Implementation error!");
		size_t datablock_view_index = static_cast<size_t>(iter.bufferView);
		assert_term(datablock_view_index<result->datablock_views.size(),"Invalid datablock view index!");
		DataBlock::View const* datablock_view = result->datablock_views[datablock_view_index];

		size_t const& offset = iter.byteOffset;

		assert_term(iter.count>0,"Implementation error!");
		size_t num_elements = iter.count;

		switch (iter.type) {
			case TINYGLTF_TYPE_SCALAR:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor< uint8_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_BYTE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<  int8_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<uint16_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_SHORT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor< int16_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<uint32_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor< int32_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<float   >( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<double  >( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_VEC2:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec2u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec2>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_VEC3:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec3u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec3>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_VEC4:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec4u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec4>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_MAT2:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   notimpl;
					case TINYGLTF_COMPONENT_TYPE_INT:            notimpl;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat2x2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat2x2>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_MAT3:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   notimpl;
					case TINYGLTF_COMPONENT_TYPE_INT:            notimpl;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat3x3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat3x3>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			case TINYGLTF_TYPE_MAT4:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  notimpl;
					case TINYGLTF_COMPONENT_TYPE_BYTE:           notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: notimpl;
					case TINYGLTF_COMPONENT_TYPE_SHORT:          notimpl;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:   notimpl;
					case TINYGLTF_COMPONENT_TYPE_INT:            notimpl;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat4x4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						result->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat4x4>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			nodefault;
		}
	}
	//	Objects (GLTF "mesh"es)
	for (tinygltf::Mesh const& iter1 : asset.meshes) {
		Object* object = new Object(iter1.name);
		result->objects.emplace_back(object);

		for (tinygltf::Primitive const& iter2 : iter1.primitives) {
			Object::Mesh* mesh;
			switch (iter2.mode) {
				case TINYGLTF_MODE_POINTS:         mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::POINTS        ); break;
				case TINYGLTF_MODE_LINE:           mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::LINES         ); break;
				case TINYGLTF_MODE_LINE_LOOP:      mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::LINE_LOOP     ); break;
				case TINYGLTF_MODE_LINE_STRIP:     mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::LINE_STRIP    ); break;
				case TINYGLTF_MODE_TRIANGLES:      mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::TRIANGLES     ); break;
				case TINYGLTF_MODE_TRIANGLE_STRIP: mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::TRIANGLE_STRIP); break;
				case TINYGLTF_MODE_TRIANGLE_FAN:   mesh=new Object::Mesh(Object::Mesh::TYPE_PRIMS::TRIANGLE_FAN  ); break;
				nodefault;
			}
			object->meshes.emplace_back(mesh);

			for (auto const& iter3 : iter2.attributes) {
				assert_term(iter3.second>=0,"Implementation error!");
				size_t datablock_accessor_index = static_cast<size_t>(iter3.second);
				assert_term(datablock_accessor_index<result->datablock_accessors.size(),"Invalid datablock accessor index!");
				DataBlock::AccessorBase const* accessor = result->datablock_accessors[datablock_accessor_index];

				if      (iter3.first=="POSITION"  ) mesh->set_ref_verts(static_cast<DataBlock::Accessor<Vec3f> const*>(accessor));
				else if (iter3.first=="NORMAL"    ) mesh->set_ref_norms(static_cast<DataBlock::Accessor<Vec3f> const*>(accessor));
				else if (iter3.first=="TANGENT"   ) mesh->set_ref_tangs(static_cast<DataBlock::Accessor<Vec4f> const*>(accessor));
				else if (iter3.first=="TEXCOORD_0") {
					switch (accessor->type) {
						case DataBlock::AccessorBase::TYPE::U8x2:  mesh->set_ref_texcoords0(static_cast<DataBlock::Accessor<Vec2ub> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::U16x2: mesh->set_ref_texcoords0(static_cast<DataBlock::Accessor<Vec2us> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::F32x2: mesh->set_ref_texcoords0(static_cast<DataBlock::Accessor<Vec2f > const*>(accessor)); break;
						nodefault;
					}
				}
				else if (iter3.first=="TEXCOORD_1") {
					switch (accessor->type) {
						case DataBlock::AccessorBase::TYPE::U8x2:  mesh->set_ref_texcoords1(static_cast<DataBlock::Accessor<Vec2ub> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::U16x2: mesh->set_ref_texcoords1(static_cast<DataBlock::Accessor<Vec2us> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::F32x2: mesh->set_ref_texcoords1(static_cast<DataBlock::Accessor<Vec2f > const*>(accessor)); break;
						nodefault;
					}
				}
				else if (iter3.first=="COLOR_0") {
					switch (accessor->type) {
						case DataBlock::AccessorBase::TYPE::U8x3:  mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec3ub> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::U16x3: mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec3us> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::F32x3: mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec3f > const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::U8x4:  mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec4ub> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::U16x4: mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec4us> const*>(accessor)); break;
						case DataBlock::AccessorBase::TYPE::F32x4: mesh->set_ref_colors0(static_cast<DataBlock::Accessor<Vec4f > const*>(accessor)); break;
						nodefault;
					}
				}
				//TODO: joints
				//TODO: weights
			}

			if (iter2.indices!=-1) {
				assert_term(iter2.indices>=0,"Implementation error!");
				size_t datablock_accessor_index = static_cast<size_t>(iter2.indices);
				assert_term(datablock_accessor_index<result->datablock_accessors.size(),"Invalid datablock accessor index!");
				DataBlock::AccessorBase const* accessor = result->datablock_accessors[datablock_accessor_index];

				switch (accessor->type) {
					case DataBlock::AccessorBase::TYPE::U16:
						mesh->set_ref_indices( static_cast<DataBlock::Accessor<uint16_t> const*>(accessor));
						break;
					case DataBlock::AccessorBase::TYPE::U32:
						mesh->set_ref_indices( static_cast<DataBlock::Accessor<uint32_t> const*>(accessor));
						break;
					nodefault;
				}
			}
		}
	}
	//	Nodes (GLTF "node"s)
	for (tinygltf::Node const& iter : asset.nodes) {
		Node* node = new Node;
		result->nodes.emplace_back(node);

		if (iter.matrix.empty()) {
			if (!iter.translation.empty()) {
				assert_term(iter.translation.size()==3,"Implementation error!");
				node->transform.type = static_cast<Node::Transform::TYPE>( node->transform.type | Node::Transform::TYPE::MSK_TRANSLATE );
				for (size_t i=0;i<3;++i) node->transform.translate[i]=static_cast<float>(iter.translation[i]);
			}
			if (!iter.rotation.empty()) {
				assert_term(iter.rotation.size()==4,"Implementation error!");
				node->transform.type = static_cast<Node::Transform::TYPE>( node->transform.type | Node::Transform::TYPE::MSK_ROTATE );
				for (size_t i=0;i<4;++i) node->transform.rotate[i]=static_cast<float>(iter.rotation[i]);
			}
			if (!iter.scale.empty()) {
				assert_term(iter.scale.size()==3,"Implementation error!");
				node->transform.type = static_cast<Node::Transform::TYPE>( node->transform.type | Node::Transform::TYPE::MSK_SCALE );
				for (size_t i=0;i<3;++i) node->transform.rotate[i]=static_cast<float>(iter.scale[i]);
			}
		} else {
			assert_term(iter.matrix.size()==16,"Implementation error!");
			node->transform.type = Node::Transform::TYPE::MATRIX;
			for (size_t k=0;k<16;++k) {
				node->transform.matrix[k/4][k%4] = static_cast<float>(iter.matrix[k]);
			}
		}
	}
	for (size_t i=0;i<asset.nodes.size();++i) {
		tinygltf::Node const& iter = asset.nodes[i];
		Node* node = result->nodes[i];

		for (int child_index : iter.children) {
			assert_term(child_index>=0&&static_cast<size_t>(child_index)<result->nodes.size(),"Implementation error!");
			node->children.emplace_back(result->nodes[static_cast<size_t>(child_index)]);
		}

		if (iter.mesh!=-1) {
			assert_term(iter.mesh>=0&&static_cast<size_t>(iter.mesh)<result->objects.size(),"Implementation error!");
			node->objects.emplace_back(result->objects[iter.mesh]);
		}
	}
	//	Scenes (GLTF "scene"s)
	for (tinygltf::Scene const& iter : asset.scenes) {
		Scene* scene = new Scene;
		result->scenes.emplace_back(scene);

		for (int root_index : iter.nodes) {
			assert_term(root_index>=0&&static_cast<size_t>(root_index)<result->nodes.size(),"Implementation error!");
			scene->root_nodes.emplace_back(result->nodes[static_cast<size_t>(root_index)]);
		}
	}

	return result;
}


}}
