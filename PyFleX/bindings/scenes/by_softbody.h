class by_SoftBody: public Scene
{

public:
	by_SoftBody(const char* name) :
		Scene(name),
		mRadius(0.1f),
		mRelaxationFactor(1.0f),
		mPlinth(false),
		plasticDeformation(false)
	{
		const Vec3 colorPicker[7] =
		{
			Vec3(0.0f, 0.5f, 1.0f),
			Vec3(0.797f, 0.354f, 0.000f),
			Vec3(0.000f, 0.349f, 0.173f),
			Vec3(0.875f, 0.782f, 0.051f),
			Vec3(0.01f, 0.170f, 0.453f),
			Vec3(0.673f, 0.111f, 0.000f),
			Vec3(0.612f, 0.194f, 0.394f)
		};
		memcpy(mColorPicker, colorPicker, sizeof(Vec3) * 7);
	}

	float mRadius;
	float mRelaxationFactor;
	bool mPlinth;

	Vec3 mColorPicker[7];

	struct Instance
	{
		Instance(const char* mesh) :

			mFile(mesh),
			mColor(0.5f, 0.5f, 1.0f),

			mScale(2.0f),
			mTranslation(0.0f, 1.0f, 0.0f),
			mRotation(0.0f, 0.0f, 0.0f, 1.0f),

			mClusterSpacing(1.0f),
			mClusterRadius(0.0f),
			mClusterStiffness(0.5f),

			mLinkRadius(0.0f),
			mLinkStiffness(1.0f),

			mGlobalStiffness(0.0f),

			mSurfaceSampling(0.0f),
			mVolumeSampling(4.0f),

			mSkinningFalloff(2.0f),
			mSkinningMaxDistance(100.0f),

			mClusterPlasticThreshold(0.0f),
			mClusterPlasticCreep(0.0f)
		{}

		const char* mFile;
		Vec3 mColor;

		Vec3 mScale;
		Vec3 mTranslation;
		Quat mRotation;

		float mClusterSpacing;
		float mClusterRadius;
		float mClusterStiffness;

		float mLinkRadius;
		float mLinkStiffness;

		float mGlobalStiffness;

		float mSurfaceSampling;
		float mVolumeSampling;

		float mSkinningFalloff;
		float mSkinningMaxDistance;

		float mClusterPlasticThreshold;
		float mClusterPlasticCreep;
	};

	std::vector<Instance> mInstances;

private:

	struct RenderingInstance
	{
		Mesh* mMesh;
		std::vector<int> mSkinningIndices;
		std::vector<float> mSkinningWeights;
		vector<Vec3> mRigidRestPoses;
		Vec3 mColor;
		int mOffset;
	};

	std::vector<RenderingInstance> mRenderingInstances;

	bool plasticDeformation;


public:
	virtual void AddInstance(Instance instance)
	{
		this->mInstances.push_back(instance);
	}

	char* make_path(char* full_path, std::string path) {
		strcpy(full_path, getenv("PYFLEXROOT"));
		strcat(full_path, path.c_str());
		return full_path;
	}

	void Initialize(py::array_t<float> scene_params, 
                    py::array_t<float> vertices,
                    py::array_t<int> stretch_edges,
                    py::array_t<int> bend_edges,
                    py::array_t<int> shear_edges,
                    py::array_t<int> faces,
                    int thread_idx = 0)
	{
		auto ptr = (float *) scene_params.request().ptr;

		Vec3 scale = Vec3(ptr[0], ptr[1], ptr[2]);
		Vec3 trans = Vec3(ptr[3], ptr[4], ptr[5]);
		
		float radius = ptr[6];
		mRadius = radius;
		
		float clusterSpacing = ptr[7];
		float clusterRadius = ptr[8];
		float clusterStiffness = ptr[9];

		float linkRadius = ptr[10];
		float linkStiffness = ptr[11];

		float globalStiffness = ptr[12];

		float surfaceSampling = ptr[13];
		float volumeSampling = ptr[14];

		float skinningFalloff = ptr[15];
		float skinningMaxDistance = ptr[16];

		float clusterPlasticThreshold = ptr[17];
		float clusterPlasticCreep = ptr[18];
		
		float dynamicFriction = ptr[19];
		float particleFrinction = ptr[20];
		
		int draw_mesh = (int) ptr[21];

		float relaxtion_factor = ptr[22];
		mRelaxationFactor = relaxtion_factor;

		Vec3 rotate_v = Vec3(ptr[23], ptr[24], ptr[25]);
		float rotate_w = ptr[26];
		Quat rotate = Quat(rotate_v, rotate_w);

		float collisionDistance = ptr[27];

		int fixed_particles = (int) ptr[28];
		int fixed_coord = (int) ptr[29];

		// Which object to load: the box (exact cube) or cube_mesh (rectangular)
		int obj_to_load = (int) ptr[30];

		// Damping
		float damping = ptr[31];

		// Softbody or rigid
		float stiffness = ptr[32];

		char box_path[100];
		std::string obj_path;
		if (obj_to_load == 0) {
			obj_path = "/data/box.ply";
		} else if (obj_to_load == 1) {
			obj_path = "/data/sphere.ply";
		} else if (obj_to_load == 2) {
			// Borrowed this mesh from:
			// https://github.com/fracture91/graphics-hw3/blob/master/meshes/cylinder.ply
			obj_path = "/data/cylinder.ply";
		} else {
			//obj_path = "/data/rigid/cube_mesh.ply";
			obj_path = "/data/box.ply";
		}
		std::cout << "object to be loaded: " << obj_path << std::endl;
		//Instance box(make_path(box_path, "/data/box.ply"));
        //Instance box(make_path(box_path, "/data/rigid/cube_mesh.ply"));
		Instance box(make_path(box_path, obj_path));
		box.mScale = scale;
		box.mTranslation = trans;
		box.mRotation = rotate;
		box.mClusterSpacing = clusterSpacing;
		box.mClusterRadius = clusterRadius;
		box.mClusterStiffness = clusterStiffness;
		box.mLinkRadius = linkRadius;
		box.mLinkStiffness = linkStiffness;
		box.mGlobalStiffness = globalStiffness;
		box.mSurfaceSampling = surfaceSampling;
		box.mVolumeSampling = volumeSampling;
		box.mSkinningFalloff = skinningFalloff;
		box.mSkinningMaxDistance = skinningMaxDistance;
		box.mClusterPlasticThreshold = clusterPlasticThreshold;
		box.mClusterPlasticCreep = clusterPlasticCreep;
		AddInstance(box);
        


		// no fluids or sdf based collision
		g_solverDesc.featureMode = eNvFlexFeatureModeSimpleSolids;

		g_params.radius = radius;
		g_params.dynamicFriction = dynamicFriction;
		g_params.particleFriction = particleFrinction;
		g_params.numIterations = 10; //4;
		g_params.collisionDistance = collisionDistance;

		g_params.relaxationFactor = mRelaxationFactor;
		g_params.damping = damping;

		g_windStrength = 0.0f;

		g_numSubsteps = 2;

		// draw options
		g_drawPoints = draw_mesh == 1 ? false : true;
		g_wireframe = false;
		g_drawSprings = false;
		g_drawBases = false;
		g_drawMesh = draw_mesh == 1 ? true : false;

		g_buffers->rigidOffsets.push_back(0);

		mRenderingInstances.resize(0);

		// build soft bodies 
		// for (int i = 0; i < int(mInstances.size()); i++)
		CreateSoftBody(stiffness, fixed_coord, fixed_particles, mInstances[0], mRenderingInstances.size());

		if (mPlinth) 
			AddPlinth();

		// fix any particles below the ground plane in place
		for (int i = 0; i < int(g_buffers->positions.size()); ++i)
			if (g_buffers->positions[i].y < 0.4f)
				g_buffers->positions[i].w = 0.0f;

		// expand radius for better self collision
		g_params.radius *= 1.5f;

		g_lightDistance *= 1.5f;
	}

	void CreateSoftBody(float stiffness, int fixed_coord, int fixed_particles, Instance instance, int group = 0, bool texture=false)
	{
		RenderingInstance renderingInstance;

		Mesh* mesh = ImportMesh(GetFilePathByPlatform(instance.mFile).c_str(), texture);
		mesh->Normalize();
		mesh->Transform(ScaleMatrix(instance.mScale*mRadius));
		mesh->Transform(RotationMatrix(instance.mRotation)); 
		mesh->Transform(TranslationMatrix(Point3(instance.mTranslation)));
		// mesh->Transform(TranslationMatrix(Point3(instance.mTranslation))*ScaleMatrix(instance.mScale*mRadius));
		// mesh->Transform(RotationMatrix(instance.mRotation));
		

		renderingInstance.mMesh = mesh;
		renderingInstance.mColor = instance.mColor;
		renderingInstance.mOffset = g_buffers->rigidTranslations.size();

		double createStart = GetSeconds();

		// Testing out creating rigid body definition
		NvFlexExtAsset* asset;
		if (stiffness >= 1.3) {
			std::cout << "Creating rigid from mesh..." << std::endl;
			asset = NvFlexExtCreateRigidFromMesh(
				(float*)&renderingInstance.mMesh->m_positions[0],
				renderingInstance.mMesh->m_positions.size(),
				(int*)&renderingInstance.mMesh->m_indices[0],
				renderingInstance.mMesh->m_indices.size(),
				// 0.03,
				// -0.03*0.5f
				mRadius,
				-mRadius*0.5f
			);
		} else {
			// create soft body definition
			std::cout << "TESING PYFLEX RECOMPILE CHANGES Creating soft from mesh..." << std::endl;
			asset = NvFlexExtCreateSoftFromMesh(
				(float*)&renderingInstance.mMesh->m_positions[0],
				renderingInstance.mMesh->m_positions.size(),
				(int*)&renderingInstance.mMesh->m_indices[0],
				renderingInstance.mMesh->m_indices.size(),
				mRadius,
				instance.mVolumeSampling,
				instance.mSurfaceSampling,
				instance.mClusterSpacing*mRadius,
				instance.mClusterRadius*mRadius,
				instance.mClusterStiffness,
				instance.mLinkRadius*mRadius,
				instance.mLinkStiffness,
				instance.mGlobalStiffness,
				instance.mClusterPlasticThreshold,
				instance.mClusterPlasticCreep);
		}
		std::cout << "Finished function for creating soft from mesh!" << std::endl;
		
		// create soft body definition
		// NvFlexExtAsset* asset = NvFlexExtCreateSoftFromMesh(
		// 	(float*)&renderingInstance.mMesh->m_positions[0],
		// 	renderingInstance.mMesh->m_positions.size(),
		// 	(int*)&renderingInstance.mMesh->m_indices[0],
		// 	renderingInstance.mMesh->m_indices.size(),
		// 	mRadius,
		// 	instance.mVolumeSampling,
		// 	instance.mSurfaceSampling,
		// 	instance.mClusterSpacing*mRadius,
		// 	instance.mClusterRadius*mRadius,
		// 	instance.mClusterStiffness,
		// 	instance.mLinkRadius*mRadius,
		// 	instance.mLinkStiffness,
		// 	instance.mGlobalStiffness,
		// 	instance.mClusterPlasticThreshold,
		// 	instance.mClusterPlasticCreep);

		double createEnd = GetSeconds();

		// create skinning
		const int maxWeights = 4;

		renderingInstance.mSkinningIndices.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);
		renderingInstance.mSkinningWeights.resize(renderingInstance.mMesh->m_positions.size()*maxWeights);

		for (int i = 0; i < asset->numShapes; ++i)
			renderingInstance.mRigidRestPoses.push_back(Vec3(&asset->shapeCenters[i * 3]));

		double skinStart = GetSeconds();

		NvFlexExtCreateSoftMeshSkinning(
			(float*)&renderingInstance.mMesh->m_positions[0],
			renderingInstance.mMesh->m_positions.size(),
			asset->shapeCenters,
			asset->numShapes,
			instance.mSkinningFalloff,
			instance.mSkinningMaxDistance,
			&renderingInstance.mSkinningWeights[0],
			&renderingInstance.mSkinningIndices[0]);

		double skinEnd = GetSeconds();

		printf("Created soft in %f ms Skinned in %f\n", (createEnd - createStart)*1000.0f, (skinEnd - skinStart)*1000.0f);

		const int particleOffset = g_buffers->positions.size();
		const int indexOffset = g_buffers->rigidOffsets.back();

		std::vector<float> fixed_coordinates;
		// add particle data to solver
		std::cout << "asset->numShapes:" << asset->numShapes << std::endl;
		for (int i = 0; i < asset->numParticles; ++i)
		{
			g_buffers->positions.push_back(&asset->particles[i * 4]);
			g_buffers->velocities.push_back(0.0f);
			if (fixed_coord == 0) {
				fixed_coordinates.push_back(g_buffers->positions[g_buffers->positions.size()-1].x);
			} else if (fixed_coord == 1) {
				fixed_coordinates.push_back(g_buffers->positions[g_buffers->positions.size()-1].y);
			} else {
				fixed_coordinates.push_back(g_buffers->positions[g_buffers->positions.size()-1].z);
			}

			const int phase = NvFlexMakePhase(group, eNvFlexPhaseSelfCollide | eNvFlexPhaseSelfCollideFilter);
			g_buffers->phases.push_back(phase);
		}

		// fix the particles closest to the ground (bottom layer) by setting inverse mass to 0 (mass -> inf)
		// sort the particles by y coord, determine which ones are "lowest" to set to be fixed in place
		// Get all the y-coordinates
		if (fixed_particles > 0) {
			std::sort(fixed_coordinates.begin(), fixed_coordinates.end());
			float min_fixed_coord = fixed_coordinates[0];
			float max_fixed_coord = fixed_coordinates[fixed_coordinates.size()-1];
			std::cout << "fixed coordinate: " << fixed_coord << std::endl;
			std::cout << "min coord: " << min_fixed_coord << ", max coord: " << max_fixed_coord << std::endl;
			// Set lowest 1/3 to be fixed
			float threshold = fixed_coordinates[(int)(fixed_coordinates.size() / fixed_particles)];
			std::cout << "fixed coord threshold: " << threshold << std::endl;
			// Start for the particles of this instance only
			int start = (g_buffers->positions.size() - asset->numParticles);
			std::cout << "starting index for determining which points to be fixed: " << start << std::endl;
			std::cout << "size of positions vector: " << g_buffers->positions.size() << std::endl;
			int count = 0;
			for (int i = start; i < int(g_buffers->positions.size()); ++i) {
				if ((fixed_coord == 1 && g_buffers->positions[i].y < threshold) ||
				(fixed_coord == 0 && g_buffers->positions[i].x < threshold) ||
				(fixed_coord == 2 && g_buffers->positions[i].z < threshold)) {
					g_buffers->positions[i].w = 0.0f;
					count++;
					// std::cout << "fixed particle based on calculation" << std::endl;
				}
			}
			std::cout << "num particles that are fixed: " << count << std::endl;
		}
		// add shape data to solver
		for (int i = 0; i < asset->numShapeIndices; ++i)
			g_buffers->rigidIndices.push_back(asset->shapeIndices[i] + particleOffset);

		for (int i = 0; i < asset->numShapes; ++i)
		{
			g_buffers->rigidOffsets.push_back(asset->shapeOffsets[i] + indexOffset);
			g_buffers->rigidTranslations.push_back(Vec3(&asset->shapeCenters[i * 3]));
			g_buffers->rigidRotations.push_back(Quat());
			g_buffers->rigidCoefficients.push_back(asset->shapeCoefficients[i]);
		}


		// add plastic deformation data to solver, if at least one asset has non-zero plastic deformation coefficients, leave the according pointers at NULL otherwise
		if (plasticDeformation)
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
				}
			}
			else
			{
				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(0.0f);
					g_buffers->rigidPlasticCreeps.push_back(0.0f);
				}
			}
		}
		else 
		{
			if (asset->shapePlasticThresholds && asset->shapePlasticCreeps)
			{
				int oldBufferSize = g_buffers->rigidCoefficients.size() - asset->numShapes;

				g_buffers->rigidPlasticThresholds.resize(oldBufferSize);
				g_buffers->rigidPlasticCreeps.resize(oldBufferSize);

				for (int i = 0; i < oldBufferSize; i++)
				{
					g_buffers->rigidPlasticThresholds[i] = 0.0f;
					g_buffers->rigidPlasticCreeps[i] = 0.0f;
				}

				for (int i = 0; i < asset->numShapes; ++i)
				{
					g_buffers->rigidPlasticThresholds.push_back(asset->shapePlasticThresholds[i]);
					g_buffers->rigidPlasticCreeps.push_back(asset->shapePlasticCreeps[i]);
				}

				plasticDeformation = true;
			}
		}

		// add link data to the solver 
		// std::cout << "asset->numSprings:" << asset->numSprings << std::endl;
		 std::cout << "asset->numSprings:" << asset->numSprings << std::endl;
		for (int i = 0; i < asset->numSprings; ++i)
		{
			g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 0]);
			g_buffers->springIndices.push_back(asset->springIndices[i * 2 + 1]);

			g_buffers->springStiffness.push_back(asset->springCoefficients[i]);
			g_buffers->springLengths.push_back(asset->springRestLengths[i]);
		}

		NvFlexExtDestroyAsset(asset);

		mRenderingInstances.push_back(renderingInstance);
	}

	virtual void Draw(int pass)
	{
		if (!g_drawMesh)
			return;

		for (int s = 0; s < int(mRenderingInstances.size()); ++s)
		{
			const RenderingInstance& instance = mRenderingInstances[s];

			Mesh m;
			m.m_positions.resize(instance.mMesh->m_positions.size());
			m.m_normals.resize(instance.mMesh->m_normals.size());
			m.m_indices = instance.mMesh->m_indices;

			for (int i = 0; i < int(instance.mMesh->m_positions.size()); ++i)
			{
				Vec3 softPos;
				Vec3 softNormal;

				for (int w = 0; w < 4; ++w)
				{
					const int cluster = instance.mSkinningIndices[i * 4 + w];
					const float weight = instance.mSkinningWeights[i * 4 + w];

					if (cluster > -1)
					{
						// offset in the global constraint array
						int rigidIndex = cluster + instance.mOffset;

						Vec3 localPos = Vec3(instance.mMesh->m_positions[i]) - instance.mRigidRestPoses[cluster];

						Vec3 skinnedPos = g_buffers->rigidTranslations[rigidIndex] + Rotate(g_buffers->rigidRotations[rigidIndex], localPos);
						Vec3 skinnedNormal = Rotate(g_buffers->rigidRotations[rigidIndex], instance.mMesh->m_normals[i]);

						softPos += skinnedPos*weight;
						softNormal += skinnedNormal*weight;
					}
				}

				m.m_positions[i] = Point3(softPos);
				m.m_normals[i] = softNormal;
			}

			DrawMesh(&m, instance.mColor);
		}
	}

};

