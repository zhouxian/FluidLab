#include <bindings/main.cpp>

char *make_path(char *full_path, std::string path)
{
    strcpy(full_path, getenv("FLEXRENDERERROOT"));
    strcat(full_path, path.c_str());
    return full_path;
}

Colour correct_gamma_colour(Colour color)
{
    color.r = pow(color.r, g_colorGamma);
    color.g = pow(color.g, g_colorGamma);
    color.b = pow(color.b, g_colorGamma);
    return color;
}

void renderer_init(int camera_width = 720, int camera_height = 720, int msaaSamples = 8, float fov = 0.7)
{
    g_screenWidth = camera_width;
    g_screenHeight = camera_height;
    g_msaaSamples = msaaSamples;
    g_fov = fov;


    g_pause = false;
    g_graphics = 0;

    // Create the demo context
    CreateDemoContext(g_graphics);

    RenderInitOptions options;
    options.numMsaaSamples = g_msaaSamples;

    InitRenderHeadless(options, g_screenWidth, g_screenHeight);
    g_fluidRenderer = CreateFluidRenderer(g_screenWidth, g_screenHeight);

    NvFlexInitDesc desc;
    desc.deviceIndex = g_device;
    desc.enableExtensions = g_extensions;
    desc.renderDevice = 0;
    desc.renderContext = 0;
    desc.computeContext = 0;
    desc.computeType = eNvFlexCUDA;

    // Init Flex library, note that no CUDA methods should be called before this
    // point to ensure we get the device context we want
    g_flexLib = NvFlexInit(NV_FLEX_VERSION, ErrorCallback, &desc);

    if (g_Error || g_flexLib == nullptr)
    {
        printf("Could not initialize Flex, exiting.\n");
        exit(-1);
    }

    // create shadow maps
    g_shadowMap = ShadowCreate();

}

void renderer_set_body_color(int bodyId, py::array_t<float> bodiesColor)
{
    auto ptr_bodiesColor = (float *)bodiesColor.request().ptr;
    g_bodiesColor[bodyId] = Vec4(ptr_bodiesColor[0],
                                ptr_bodiesColor[1],
                                ptr_bodiesColor[2],
                                ptr_bodiesColor[3]);
}

void renderer_create_scene(bool renderParticle, float particleRadius, float smokeRadius, float anisotropyScale, float smoothing, float colorGamma, float fluidRestDistance, py::array_t<float> lightPos, py::array_t<float> lightTarget, float lightFov, float floorHeight, float sceneRadius, py::array_t<float> bodiesNumParticles, py::array_t<float> bodiesParticleOffset, py::array_t<float> bodiesColor, py::array_t<bool> bodiesNeedsSmoothing, int nBodies)
{
    g_colorGamma = colorGamma;

    // bodies info
    auto ptr_bodiesNumParticles = (float *)bodiesNumParticles.request().ptr;
    auto ptr_bodiesParticleOffset = (float *)bodiesParticleOffset.request().ptr;
    auto ptr_bodiesNeedsSmoothing = (bool *)bodiesNeedsSmoothing.request().ptr;
    auto ptr_bodiesColor = (float *)bodiesColor.request().ptr;
    for (int i = 0; i < nBodies; i++)
    {
        g_bodiesNumParticles.push_back(ptr_bodiesNumParticles[i]);
        g_bodiesParticleOffset.push_back(ptr_bodiesParticleOffset[i]);
        g_bodiesNeedsSmoothing.push_back(ptr_bodiesNeedsSmoothing[i]);
        g_bodiesColor.push_back(Vec4(ptr_bodiesColor[i * 4],
                                    ptr_bodiesColor[i * 4 + 1],
                                    ptr_bodiesColor[i * 4 + 2],
                                    ptr_bodiesColor[i * 4 + 3]));

    }

    Init();

    if (renderParticle == true) {
        g_drawDensity = false;
        g_drawDiffuse = false;
        g_drawEllipsoids = false;
        g_drawPoints = true;
    }
    else {
        g_drawDensity = false;
        g_drawDiffuse = false;
        g_drawEllipsoids = true;
        g_drawPoints = false;
    }

    g_particleRadius           = particleRadius;
    g_smokeRadius              = smokeRadius;
    g_params.anisotropyScale   = anisotropyScale;
    g_params.smoothing         = smoothing;
    g_params.fluidRestDistance = fluidRestDistance;
    g_params.planes[0][3]      = -floorHeight;
    g_params.planes[1][3]      = sceneRadius;
    g_params.planes[2][3]      = sceneRadius;
    g_params.planes[3][3]      = sceneRadius;
    g_params.planes[4][3]      = sceneRadius;
    g_params.planes[5][3]      = sceneRadius;


    auto ptr_lightPos = (float *)lightPos.request().ptr;
    g_lightPos = Vec3(ptr_lightPos[0], ptr_lightPos[1], ptr_lightPos[2]);
    auto ptr_lightTarget = (float *)lightTarget.request().ptr;
    g_lightTarget = Vec3(ptr_lightTarget[0], ptr_lightTarget[1], ptr_lightTarget[2]);

    g_lightFov = DegToRad(lightFov);

}

void renderer_clean()
{
    if (g_fluidRenderer)
        DestroyFluidRenderer(g_fluidRenderer);
    if (g_fluidRenderBuffers)
        DestroyFluidRenderBuffers(g_fluidRenderBuffers);
    if (g_diffuseRenderBuffers)
        DestroyDiffuseRenderBuffers(g_diffuseRenderBuffers);
    if (g_shadowMap)
        ShadowDestroy(g_shadowMap);
    Shutdown();
}

int main()
{
    renderer_init();
    renderer_clean();

    return 0;
}


float rand_float(float LO, float HI)
{
    return LO + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (HI - LO)));
}


int renderer_get_n_particles()
{
    g_buffers->positions.map();
    int n_particles = g_buffers->positions.size();
    g_buffers->positions.unmap();
    return n_particles;
}

py::array_t<float> renderer_get_positions()
{
    g_buffers->positions.map();
    auto positions = py::array_t<float>((size_t)g_buffers->positions.size() * 3);
    auto ptr = (float *)positions.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->positions.size(); i++)
    {
        ptr[i * 3] = g_buffers->positions[i].x;
        ptr[i * 3 + 1] = g_buffers->positions[i].y;
        ptr[i * 3 + 2] = g_buffers->positions[i].z;
    }

    g_buffers->positions.unmap();

    return positions;
}

void renderer_set_particles_state(py::array_t<float> positions, py::array_t<int> used)
{
    auto positions_ptr = (float *)positions.request().ptr;
    auto used_ptr = (int *)used.request().ptr;


    g_buffers->positions.map();

    for (size_t i = 0; i < (size_t)g_buffers->positions.size(); i++)
    {
        if (used_ptr[i]) {

            g_buffers->positions[i] = Vec4(positions_ptr[i * 3], positions_ptr[i * 3 + 1], positions_ptr[i * 3 + 2], 1.0);
        }
        // for unused particles, put them at far away with enough spacing in between, to reduce flex solver load
        else {
            g_buffers->positions[i] = Vec4(1000.0 + i, 1000.0, 1000.0, 1.0);
        }

    }

    g_buffers->positions.unmap();
}

void renderer_add_smoke_particles(int particles_size, py::array_t<float> positions, py::array_t<float> colors)
{
    Point3 *vertices_new = new Point3[particles_size];
    Colour *colors_new   = new Colour[particles_size];
    int *indices_new     = new int[particles_size];

    g_smoke = new SmokeParticles();
    g_smoke->particles_size = particles_size;
    g_smoke->p_positions.assign(vertices_new, vertices_new + particles_size);
    g_smoke->p_colours.assign(colors_new, colors_new + particles_size);
    g_smoke->p_indices.assign(indices_new, indices_new + particles_size);
    for (size_t i = 0; i < (size_t)particles_size; i++)
    {
        g_smoke->p_indices[i] = i;
    }
    gpu_smoke = CreateGpuSmoke(g_smoke);

    g_smoke->p_positions_raw = positions;
    g_smoke->p_colours_raw = colors;
    UpdateGpuSmoke(gpu_smoke, g_smoke, true);
}

void renderer_update_smoke_particles(py::array_t<float> colors)
{
    g_smoke->p_colours_raw = colors;
}

int renderer_add_mesh(int vertices_size, int indices_size, py::array_t<float> colors, py::array_t<int> indices)
{

    Point3 *vertices_new = new Point3[vertices_size];
    Vec3 *normals_new    = new Vec3[vertices_size];
    Colour *colors_new   = new Colour[vertices_size];
    int *indices_new     = new int[indices_size];

    auto ptr_color = (float *)colors.request().ptr;
    auto ptr_ind = (int *)indices.request().ptr;

    for (size_t i = 0; i < (size_t)vertices_size; i++)
    {
        colors_new[i] = correct_gamma_colour(Colour(ptr_color[i * 4], ptr_color[i * 4 + 1], ptr_color[i * 4 + 2], ptr_color[i * 4 + 3]));;
    }

    for (size_t i = 0; i < (size_t)indices_size; i++)
    {
        indices_new[i] = ptr_ind[i];
    }

    Mesh *m = new Mesh();
    m->m_positions.assign(vertices_new, vertices_new + vertices_size);
    m->m_normals.assign(normals_new, normals_new + vertices_size);
    m->m_colours.assign(colors_new, colors_new + vertices_size);
    m->m_indices.assign(indices_new, indices_new + indices_size);

    g_meshList.push_back(m);

    delete[] vertices_new;
    delete[] normals_new;
    delete[] indices_new;
    delete[] colors_new;

    return g_meshList.size() - 1;
}

int renderer_update_mesh(int mesh_id, py::array_t<float> vertices, py::array_t<float> normals)
{
    auto ptr_vert = (float *)vertices.request().ptr;
    auto ptr_normal = (float *)normals.request().ptr;
    
    for (size_t i = 0; i < (size_t)g_meshList[mesh_id]->m_positions.size(); i++)
    {
        g_meshList[mesh_id]->m_positions[i] = Point3(ptr_vert[i * 3], ptr_vert[i * 3 + 1], ptr_vert[i * 3 + 2]);
        g_meshList[mesh_id]->m_normals[i] = Vec3(ptr_normal[i * 3], ptr_normal[i * 3 + 1], ptr_normal[i * 3 + 2]);
    }
}

py::array_t<float> renderer_get_anisotropy1()
{
    g_buffers->anisotropy1.map();
    auto anisotropy1 = py::array_t<float>((size_t)g_buffers->anisotropy1.size() * 4);
    auto ptr = (float *)anisotropy1.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->anisotropy1.size(); i++)
    {
        ptr[i * 4] = g_buffers->anisotropy1[i].x;
        ptr[i * 4 + 1] = g_buffers->anisotropy1[i].y;
        ptr[i * 4 + 2] = g_buffers->anisotropy1[i].z;
        ptr[i * 4 + 3] = g_buffers->anisotropy1[i].w;
    }

    g_buffers->anisotropy1.unmap();

    return anisotropy1;
}

py::array_t<float> renderer_get_anisotropy2()
{
    g_buffers->anisotropy2.map();
    auto anisotropy2 = py::array_t<float>((size_t)g_buffers->anisotropy2.size() * 4);
    auto ptr = (float *)anisotropy2.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->anisotropy2.size(); i++)
    {
        ptr[i * 4] = g_buffers->anisotropy2[i].x;
        ptr[i * 4 + 1] = g_buffers->anisotropy2[i].y;
        ptr[i * 4 + 2] = g_buffers->anisotropy2[i].z;
        ptr[i * 4 + 3] = g_buffers->anisotropy2[i].w;
    }

    g_buffers->anisotropy2.unmap();

    return anisotropy2;
}

py::array_t<float> renderer_get_anisotropy3()
{
    g_buffers->anisotropy3.map();
    auto anisotropy3 = py::array_t<float>((size_t)g_buffers->anisotropy3.size() * 4);
    auto ptr = (float *)anisotropy3.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->anisotropy3.size(); i++)
    {
        ptr[i * 4] = g_buffers->anisotropy3[i].x;
        ptr[i * 4 + 1] = g_buffers->anisotropy3[i].y;
        ptr[i * 4 + 2] = g_buffers->anisotropy3[i].z;
        ptr[i * 4 + 3] = g_buffers->anisotropy3[i].w;
    }

    g_buffers->anisotropy3.unmap();

    return anisotropy3;
}

py::array_t<float> renderer_get_smoothed_positions()
{
    g_buffers->smoothPositions.map();
    auto smoothPositions = py::array_t<float>((size_t)g_buffers->smoothPositions.size() * 4);
    auto ptr = (float *)smoothPositions.request().ptr;

    for (size_t i = 0; i < (size_t)g_buffers->smoothPositions.size(); i++)
    {
        ptr[i * 4] = g_buffers->smoothPositions[i].x;
        ptr[i * 4 + 1] = g_buffers->smoothPositions[i].y;
        ptr[i * 4 + 2] = g_buffers->smoothPositions[i].z;
        ptr[i * 4 + 3] = g_buffers->smoothPositions[i].w;
    }

    g_buffers->smoothPositions.unmap();

    return smoothPositions;
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<int>, py::float_> renderer_get_camera_params()
{
    auto camPos = py::array_t<float>(3);
    auto camPosPtr = (float *)camPos.request().ptr;
    camPosPtr[0] = g_camPos.x;
    camPosPtr[1] = g_camPos.y;
    camPosPtr[2] = g_camPos.z;

    auto camAngle = py::array_t<float>(3);
    auto camAnglePtr = (float *)camAngle.request().ptr;
    camAnglePtr[0] = g_camAngle.x;
    camAnglePtr[1] = g_camAngle.y;
    camAnglePtr[2] = g_camAngle.z;

    auto camSize = py::array_t<int>(2);
    auto camSizePtr = (int *)camSize.request().ptr;
    camSizePtr[0] = g_screenWidth;
    camSizePtr[1] = g_screenHeight;

    return std::make_tuple(camPos, camAngle, camSize, py::float_(g_fov));
}


void renderer_set_camera_params(py::array_t<float> camera_pos, py::array_t<float> camera_angle, float camera_near, float camera_far)
{
    auto ptr_camera_pos = (float *)camera_pos.request().ptr;
    auto ptr_camera_angle = (float *)camera_angle.request().ptr;

    g_camPos = Vec3(ptr_camera_pos[0], ptr_camera_pos[1], ptr_camera_pos[2]);
    g_camAngle = Vec3(ptr_camera_angle[0], ptr_camera_angle[1], ptr_camera_angle[2]);

    g_camNear = camera_near;
    g_camFar = camera_far;
}

// std::tuple<
//     py::array_t<unsigned char>,
//     py::array_t<float>>
py::array_t<unsigned char> renderer_render()
{
    const int numParticles = NvFlexGetActiveCount(g_solver);

    if (numParticles)
    {

        // tick solver to give smoothed position and anisotropy
        NvFlexSetParticles(g_solver, g_buffers->positions.buffer, nullptr);
        NvFlexSetVelocities(g_solver, g_buffers->velocities.buffer, nullptr);
        NvFlexSetPhases(g_solver, g_buffers->phases.buffer, nullptr);
        NvFlexSetActive(g_solver, g_buffers->activeIndices.buffer, nullptr);
        NvFlexSetActiveCount(g_solver, g_buffers->activeIndices.size());

        NvFlexSetParams(g_solver, &g_params);
        NvFlexUpdateSolver(g_solver, g_dt, g_numSubsteps, g_profile);

        // if not using interop then we read back fluid data to host
        if (g_drawEllipsoids)
        {
            NvFlexGetSmoothParticles(g_solver, g_buffers->smoothPositions.buffer, nullptr);
            NvFlexGetAnisotropy(g_solver, g_buffers->anisotropy1.buffer, g_buffers->anisotropy2.buffer,
                                g_buffers->anisotropy3.buffer, NULL);
        }

        // read back diffuse data to host
        if (g_drawDensity)
            NvFlexGetDensities(g_solver, g_buffers->densities.buffer, nullptr);

        if (GetNumDiffuseRenderParticles(g_diffuseRenderBuffers))
        {
            NvFlexGetDiffuseParticles(g_solver, g_buffers->diffusePositions.buffer, g_buffers->diffuseVelocities.buffer,
                                      g_buffers->diffuseCount.buffer);
        }

    }


    MapBuffers(g_buffers);

    // // revert smoothing for nonfluid particles
    // for (size_t i = 0; i < (size_t)g_bodiesNumParticles.size(); i++)
    // {
    //     if (!g_bodiesNeedsSmoothing[i])
    //     {
    //         for (int j = 0; j < g_bodiesNumParticles[i]; j++)
    //         {
    //             int id = g_bodiesParticleOffset[i] + j;

    //             g_buffers->anisotropy1[id].x = 1.0;
    //             g_buffers->anisotropy1[id].y = 0.0;
    //             g_buffers->anisotropy1[id].z = 0.0;
    //             g_buffers->anisotropy1[id].w = g_particleRadius;
    //             g_buffers->anisotropy2[id].x = 0.0;
    //             g_buffers->anisotropy2[id].y = 1.0;
    //             g_buffers->anisotropy2[id].z = 0.0;
    //             g_buffers->anisotropy2[id].w = g_particleRadius;
    //             g_buffers->anisotropy3[id].x = 0.0;
    //             g_buffers->anisotropy3[id].y = 0.0;
    //             g_buffers->anisotropy3[id].z = 1.0;
    //             g_buffers->anisotropy3[id].w = g_particleRadius;

    //             g_buffers->smoothPositions[id].x = g_buffers->positions[id].x;
    //             g_buffers->smoothPositions[id].y = g_buffers->positions[id].y;
    //             g_buffers->smoothPositions[id].z = g_buffers->positions[id].z;
    //             g_buffers->smoothPositions[id].w = g_buffers->positions[id].w;
    //         }
    //     }
    // }


    StartFrame(Vec4(g_clearColor, 1.0f));

    RenderScene();

    EndFrame();

    int *rendered_img_int32_ptr = new int[g_screenWidth * g_screenHeight];
    ReadFrame(rendered_img_int32_ptr, g_screenWidth, g_screenHeight);

    auto rendered_img = py::array_t<uint8_t>((int)g_screenWidth * g_screenHeight * 4);
    auto rendered_img_ptr = (uint8_t *)rendered_img.request().ptr;
    
    // float *rendered_depth_float_ptr = new float[g_screenWidth * g_screenHeight];
    // ReadDepth(rendered_depth_float_ptr, g_screenWidth, g_screenHeight);
    // auto rendered_depth = py::array_t<float>((float)g_screenWidth * g_screenHeight);
    // auto rendered_depth_ptr = (float *)rendered_depth.request().ptr;

    for (int i = 0; i < g_screenWidth * g_screenHeight; ++i)
    {
        int32_abgr_to_int8_rgba((uint32_t)rendered_img_int32_ptr[i],
                                rendered_img_ptr[4 * i],
                                rendered_img_ptr[4 * i + 1],
                                rendered_img_ptr[4 * i + 2],
                                rendered_img_ptr[4 * i + 3]);
        // rendered_depth_ptr[i] = 2 * g_camFar * g_camNear / (g_camFar + g_camNear - (2 * rendered_depth_float_ptr[i] - 1) * (g_camFar - g_camNear));
    }

    delete[] rendered_img_int32_ptr;
    // delete[] rendered_depth_float_ptr;

    UnmapBuffers(g_buffers);

    return rendered_img;

    // return std::make_tuple(
    //     rendered_img,
    //     rendered_depth);

}

PYBIND11_MODULE(flex_renderer, m)
{
    m.def("main", &main);
    m.def("init", &renderer_init);
    m.def("create_scene", &renderer_create_scene);
    m.def("set_body_color", &renderer_set_body_color);
    m.def("clean", &renderer_clean);
    m.def("render", &renderer_render);
    m.def("get_camera_params", &renderer_get_camera_params);
    m.def("set_camera_params", &renderer_set_camera_params);
    m.def("get_n_particles", &renderer_get_n_particles);
    m.def("get_positions", &renderer_get_positions);
    m.def("set_particles_state", &renderer_set_particles_state);
    m.def("get_smoothed_positions", &renderer_get_smoothed_positions);
    m.def("get_anisotropy1", &renderer_get_anisotropy1);
    m.def("get_anisotropy2", &renderer_get_anisotropy2);
    m.def("get_anisotropy3", &renderer_get_anisotropy3);
    m.def("add_mesh", &renderer_add_mesh);
    m.def("update_mesh", &renderer_update_mesh);
    m.def("add_smoke_particles", &renderer_add_smoke_particles);
    m.def("update_smoke_particles", &renderer_update_smoke_particles);
}
