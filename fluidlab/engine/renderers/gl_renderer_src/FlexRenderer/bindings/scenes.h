#pragma once
#include <iostream>
#include <vector>

class Scene
{
public:
	Scene(const char *name) : mName(name) {}

	virtual void Initialize(py::dict scene_params){};
	virtual void PostInitialize() {}

	// update any buffers (all guaranteed to be mapped here)
	virtual void Update(py::array_t<float> update_params) {}

	// send any changes to flex (all buffers guaranteed to be unmapped here)
	virtual void Sync() {}

	virtual void Draw(int pass) {}
	virtual void KeyDown(int key) {}
	virtual void DoGui() {}
	virtual void CenterCamera() {}

	virtual Matrix44 GetBasis() { return Matrix44::kIdentity; }

	virtual const char *GetName() { return mName; }

	const char *mName;
};


class EmptyScene : public Scene
{
public:

    EmptyScene(const char *name) : Scene(name) {}

    void Initialize()
    {
        float radius = 0.1f;

        g_sceneLower = Vec3(-1.0f);
        g_sceneUpper = Vec3(1.0f);


        // create particles
        Vec3 velocity = Vec3(0.0f);
        int phase_fluid = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide | eNvFlexPhaseFluid);
        int phase_nonfluid = NvFlexMakePhase(0, eNvFlexPhaseSelfCollide);
        for (size_t i = 0; i < (size_t)g_bodiesNumParticles.size(); i++)
        {
            for (int j = 0; j < g_bodiesNumParticles[i]; ++j)
            {
                Vec3 position = Vec3(0.0f, 0.0f, -0.0f);
                g_buffers->positions.push_back(Vec4(position.x, position.y, position.z, 1.0f));
                g_buffers->velocities.push_back(velocity);
                if (g_bodiesNeedsSmoothing[i])
                {
                    g_buffers->phases.push_back(phase_fluid);
                }
                else
                {
                    g_buffers->phases.push_back(phase_nonfluid);
                }
            }
        }


        float restDistance = 0.02f;

        g_params.radius = radius;
        g_params.dynamicFriction = 0.00f;
        g_params.viscosity =  0.0; //2.0f;
        g_params.numIterations = 1;
        g_params.vorticityConfinement = 0.0f;
        g_params.fluidRestDistance = restDistance;
        g_params.solidPressure = 0.f;
        g_params.relaxationFactor = 0.0f;
        g_params.collisionDistance = 0.000f;
        g_params.cohesion = 0.00f;
        g_params.maxAcceleration = 0.0f;
        g_dt = 0.0001;
        g_numSubsteps = 1;

        g_maxDiffuseParticles = 0;
        g_diffuseScale = 0.5f;


    }
};