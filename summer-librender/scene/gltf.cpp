#include "gltf.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "materials/image.hpp"
#include "materials/material.hpp"
#include "materials/sampler.hpp"

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
	SceneGraph* scenegraph = new SceneGraph;
	//	Datablocks (GLTF "buffer"s)
	for (tinygltf::Buffer const& iter : asset.buffers) {
		scenegraph->datablocks.emplace_back(new DataBlock(
			iter.data
		));
	}
	//	Datablock views (GLTF "buffer view"s)
	for (tinygltf::BufferView const& iter : asset.bufferViews) {
		assert_term(iter.buffer>=0,"Implementation error!");
		size_t datablock_index = static_cast<size_t>(iter.buffer);
		assert_term(datablock_index<scenegraph->datablocks.size(),"Invalid datablock index!");

		scenegraph->datablock_views.emplace_back(new DataBlock::View(
			scenegraph->datablocks[datablock_index],
			iter.byteOffset, iter.byteStride, iter.byteLength
		));
	}
	//	Datablock accessors (GLTF "buffer accessor"s)
	for (tinygltf::Accessor const& iter : asset.accessors) {
		assert_term(iter.bufferView>=0,"Implementation error!");
		size_t datablock_view_index = static_cast<size_t>(iter.bufferView);
		assert_term(datablock_view_index<scenegraph->datablock_views.size(),"Invalid datablock view index!");
		DataBlock::View const* datablock_view = scenegraph->datablock_views[datablock_view_index];

		size_t const& offset = iter.byteOffset;

		assert_term(iter.count>0,"Implementation error!");
		size_t num_elements = iter.count;

		switch (iter.type) {
			case TINYGLTF_TYPE_SCALAR:
				switch (iter.componentType) {
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor< uint8_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_BYTE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<  int8_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<uint16_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_SHORT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor< int16_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<uint32_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor< int32_t>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<float   >( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<double  >( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec2u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec2>( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec3u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec3>( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<Vec4u>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_INT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::ivec4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_FLOAT:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: vec4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dvec4>( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat2x2>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat2x2>( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat3x3>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat3x3>( datablock_view, offset,num_elements )); break;
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
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm:: mat4x4>( datablock_view, offset,num_elements )); break;
					case TINYGLTF_COMPONENT_TYPE_DOUBLE:
						scenegraph->datablock_accessors.emplace_back(new DataBlock::Accessor<glm::dmat4x4>( datablock_view, offset,num_elements )); break;
					nodefault;
				}
				break;
			nodefault;
		}
	}
	//	Samplers (GLTF "sampler"s)
	for (tinygltf::Sampler const& iter : asset.samplers) {
		Sampler* sampler = new Sampler;
		scenegraph->samplers.emplace_back(sampler);

		switch (iter.minFilter) {
			case -1: [[fallthrough]];
			case GL_NEAREST:                sampler->type_filter=Sampler::TYPE_FILTER::MIP_NONE_TEX_CLOSEST;    break;
			case GL_LINEAR:                 sampler->type_filter=Sampler::TYPE_FILTER::MIP_NONE_TEX_LINEAR;     break;
			case GL_NEAREST_MIPMAP_NEAREST: sampler->type_filter=Sampler::TYPE_FILTER::MIP_CLOSEST_TEX_CLOSEST; break;
			case GL_LINEAR_MIPMAP_NEAREST:  sampler->type_filter=Sampler::TYPE_FILTER::MIP_CLOSEST_TEX_LINEAR;  break;
			case GL_NEAREST_MIPMAP_LINEAR:  sampler->type_filter=Sampler::TYPE_FILTER::MIP_LINEAR_TEX_CLOSEST;  break;
			case GL_LINEAR_MIPMAP_LINEAR:   sampler->type_filter=Sampler::TYPE_FILTER::MIP_LINEAR_TEX_LINEAR;   break;
			nodefault;
		}
		#ifdef BUILD_DEBUG
		switch (iter.magFilter) {
			case -1: [[fallthrough]];
			case GL_NEAREST: assert_warn(
				(static_cast<uint32_t>(sampler->type_filter) & static_cast<uint32_t>(Sampler::TYPE_FILTER::MSK_TEX_CLOSEST) )!=0u,
				"Ignoring magnification filter (texture filtering mode must match minification)!"
			); break;
			case GL_LINEAR: assert_warn(
				(static_cast<uint32_t>(sampler->type_filter) & static_cast<uint32_t>(Sampler::TYPE_FILTER::MSK_TEX_LINEAR ) )!=0u,
				"Ignoring magnification filter (texture filtering mode must match minification)!"
			); break;
			nodefault;
		}
		#endif

		switch (iter.wrapS) {
			case GL_CLAMP_TO_EDGE:   sampler->type_edges.s=Sampler::TYPE_EDGE::CLAMP;         break;
			case GL_REPEAT:          sampler->type_edges.s=Sampler::TYPE_EDGE::REPEAT;        break;
			case GL_MIRRORED_REPEAT: sampler->type_edges.s=Sampler::TYPE_EDGE::REPEAT_MIRROR; break;
			nodefault;
		}
		switch (iter.wrapT) {
			case GL_CLAMP_TO_EDGE:   sampler->type_edges.t=Sampler::TYPE_EDGE::CLAMP;         break;
			case GL_REPEAT:          sampler->type_edges.t=Sampler::TYPE_EDGE::REPEAT;        break;
			case GL_MIRRORED_REPEAT: sampler->type_edges.t=Sampler::TYPE_EDGE::REPEAT_MIRROR; break;
			nodefault;
		}
	}
	scenegraph->samplers.emplace_back(new Sampler);
	//	Images (GLTF "image"s)
	for (tinygltf::Image const& iter : asset.images) {
		Image2D::FORMAT format;
		switch (iter.component) {
			case 3: format=Image2D::FORMAT::sRGB8;    break;
			case 4: format=Image2D::FORMAT::sRGB8_A8; break;
			nodefault;
		}
		assert_term(iter.bits==8,"Not implemented!");
		assert_term(iter.pixel_type==TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE,"Not implemented!");
		assert_term(!iter.as_is,"Not implemented!");

		assert_term(iter.width>0&&iter.height>0,"Invalid image dimensions!");
		Vec2zu res = Vec2zu(static_cast<size_t>(iter.width),static_cast<size_t>(iter.height));

		Image2D* image = new Image2D(format,res);
		scenegraph->images.emplace_back(image);

		image->data.cpu = iter.image;
	}
	//	Textures (GLTF "texture"s)
	for (tinygltf::Texture const& iter : asset.textures) {
		assert_term(iter.source>=0,"Implementation error!");
		size_t index_image = static_cast<size_t>(iter.source );
		assert_term(index_image<scenegraph->images.size(),"Invalid image index!");
		Image2D const* image = scenegraph->images[index_image];

		size_t index_sampler;
		if (iter.sampler!=-1) {
			index_sampler = static_cast<size_t>(iter.sampler);
			assert_term(index_sampler<scenegraph->samplers.size()-1,"Invalid sampler index!");
		} else {
			index_sampler = scenegraph->samplers.size() - 1; //default sampler
		}
		Sampler const* sampler = scenegraph->samplers[index_sampler];

		Texture2D* texture = new Texture2D(image->res);
		scenegraph->textures.emplace_back(texture);
		texture->set_gpu_cudaoptix(image,sampler);
	}
	//	Materials (GLTF "material"s)
	for (tinygltf::Material const& iter : asset.materials) {
		MaterialMetallicRoughnessRGBA* material = new MaterialMetallicRoughnessRGBA(iter.name);
		scenegraph->materials.emplace_back(material);

		material->base_color.factor = Vec4f(
			iter.pbrMetallicRoughness.baseColorFactor[0],
			iter.pbrMetallicRoughness.baseColorFactor[1],
			iter.pbrMetallicRoughness.baseColorFactor[2],
			iter.pbrMetallicRoughness.baseColorFactor[3]
		);
		if (iter.pbrMetallicRoughness.baseColorTexture.index!=-1) {
			material->base_color.texture = scenegraph->textures[iter.pbrMetallicRoughness.baseColorTexture.index];
		}

		material->metallic. factor = static_cast<float>( iter.pbrMetallicRoughness.metallicFactor  );
		material->roughness.factor = static_cast<float>( iter.pbrMetallicRoughness.roughnessFactor );
		if (iter.pbrMetallicRoughness.metallicRoughnessTexture.index!=-1) {
			material->metallic. texture = scenegraph->textures[iter.pbrMetallicRoughness.metallicRoughnessTexture.index];
			material->roughness.texture = scenegraph->textures[iter.pbrMetallicRoughness.metallicRoughnessTexture.index];
		}

		if (iter.emissiveFactor.empty()) {
			//It's not supposed to be.  TODO: tell tinygltf about this bug?
			material->emission.factor = Vec3f(0,0,0);
		} else {
			material->emission.factor = Vec3f(
				iter.emissiveFactor[0],
				iter.emissiveFactor[1],
				iter.emissiveFactor[2]
			);
		}
		if (iter.emissiveTexture.index!=-1) {
			material->emission.texture = scenegraph->textures[iter.emissiveTexture.index];
		}

		if (iter.normalTexture.index!=-1) {
			material->normal.texture = scenegraph->textures[iter.normalTexture.index];
		}
	}
	//	Objects (GLTF "mesh"es)
	for (tinygltf::Mesh const& iter1 : asset.meshes) {
		Object* object = new Object(iter1.name);
		scenegraph->objects.emplace_back(object);

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
				assert_term(datablock_accessor_index<scenegraph->datablock_accessors.size(),"Invalid datablock accessor index!");
				DataBlock::AccessorBase const* accessor = scenegraph->datablock_accessors[datablock_accessor_index];

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
				assert_term(datablock_accessor_index<scenegraph->datablock_accessors.size(),"Invalid datablock accessor index!");
				DataBlock::AccessorBase const* accessor = scenegraph->datablock_accessors[datablock_accessor_index];

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

			mesh->material       = scenegraph->materials[iter2.material];
			mesh->material_index = iter2.material;
		}
	}
	//	Nodes (GLTF "node"s)
	for (tinygltf::Node const& iter : asset.nodes) {
		Node* node = new Node;
		scenegraph->nodes.emplace_back(node);

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
				for (size_t i=0;i<3;++i) node->transform.scale[i]=static_cast<float>(iter.scale[i]);
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
		Node* node = scenegraph->nodes[i];

		for (int child_index : iter.children) {
			assert_term(child_index>=0&&static_cast<size_t>(child_index)<scenegraph->nodes.size(),"Implementation error!");
			node->children.emplace_back(scenegraph->nodes[static_cast<size_t>(child_index)]);
		}

		if (iter.mesh!=-1) {
			assert_term(iter.mesh>=0&&static_cast<size_t>(iter.mesh)<scenegraph->objects.size(),"Implementation error!");
			node->objects.emplace_back(scenegraph->objects[iter.mesh]);
		}
	}
	//	Scenes (GLTF "scene"s)
	for (tinygltf::Scene const& iter : asset.scenes) {
		Scene* scene = new Scene(scenegraph);
		scenegraph->scenes.emplace_back(scene);

		for (int root_index : iter.nodes) {
			assert_term(root_index>=0&&static_cast<size_t>(root_index)<scenegraph->nodes.size(),"Implementation error!");
			scene->root_nodes.emplace_back(scenegraph->nodes[static_cast<size_t>(root_index)]);
		}
	}

	return scenegraph;
}


}}
