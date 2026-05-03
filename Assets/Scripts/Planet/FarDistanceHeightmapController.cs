using UnityEngine;
using UnityEngine.Rendering;

/// <summary>
/// Renders a distant terrain ring using a moving grid displaced by a runtime heightmap.
/// The mesh stays centered on the player while UV sampling remains in world space.
/// Depth is skybox-style (ZWrite off + clip Z at far plane); octrees draw after and occlude by depth.
/// </summary>
public class FarDistanceHeightmapController : MonoBehaviour
{
    private const float HugeBoundsExtent = 1e6f;

    [Header("Runtime")]
    public bool enabledSystem = true;
    public Transform priority;

    [Header("Mesh")]
    [Min(32f)] public float gridWorldSize = 12000f;
    [Range(32, 512)] public int gridResolution = 192;

    [Header("Fade")]
    [Min(0f)] public float fadeStartDistance = 1800f;
    [Min(0f)] public float fadeEndDistance = 3600f;
    [Min(0f)] public float farFadeStartDistance = 20000f;
    [Min(0f)] public float farFadeEndDistance = 28000f;
    [Tooltip("Full 3D distance from player: inside inner radius visibility is 0 (hole); ramps to full by outer radius.")]
    [Min(0f)] public float playerHoleInnerRadius = 400f;
    [Min(1f)] public float playerHoleOuterRadius = 3600f;
    [Tooltip("When on, fades far only when player is deep below OctreeGrid.minSubsurfaceHeight (bedrock proxy), not vs far-noise height.")]
    public bool deepUndergroundFarFade = false;
    [Tooltip("Meters below bedrock Y before fade begins (only if deepUndergroundFarFade).")]
    [Min(0f)] public float undergroundFadeStartM = 32f;
    [Tooltip("Meters below bedrock Y where far is fully faded.")]
    [Min(1f)] public float undergroundFadeEndM = 400f;
    [Range(0f, 1f)] public float fogHorizonMix = 0.65f;
    public Color horizonFogColor = new Color(0.55f, 0.58f, 0.62f, 1f);

    [Header("Background depth (skybox-style)")]
    [Tooltip("Reversed-Z: z = w * epsilon. Classic: z = w * (1 - epsilon). Tune if horizon fights skybox.")]
    [Min(1e-7f)] public float backgroundClipEpsilon = 1e-5f;

    [Header("Planet curvature (fake sphere)")]
    [Tooltip("Sphere radius R for y -= R - sqrt(R*R - dst^2) in XZ from reference; 0 disables.")]
    [Min(0f)] public float planetCurvatureRadius = 0f;
    [Tooltip("If true, horizontal distance uses OctreeGrid.playerCamera or Camera.main; if false, uses Priority position.")]
    public bool curvatureReferenceUsesCamera = true;

    [Header("Look")]
    public Color farColor = new Color(0.40f, 0.42f, 0.45f, 1f);
    public Material materialOverride;
    [Tooltip("Optional world-space Y offset on sampled height.")]
    [Min(0f)] public float underlayYOffset = 0f;

    [Header("Heightmap")]
    [Range(128, 2048)] public int heightTextureResolution = 1024;
    [Min(0.02f)] public float heightRefreshSeconds = 0.75f;
    public bool periodicHeightRefresh = false;
    public ComputeShader computeShaderOverride;

    private GameObject farRootObject;
    private GameObject farMeshObject;
    private MeshFilter farMeshFilter;
    private MeshRenderer farMeshRenderer;
    private Mesh farMesh;
    private Material runtimeMaterial;
    private FarDistanceHeightmapGenerator generator;

    private int meshResolutionBuilt = -1;

    public void EnsureInitialized()
    {
        if (farMeshObject == null)
            CreateFarMeshObject();

        if (generator == null)
            generator = GetComponent<FarDistanceHeightmapGenerator>() ?? gameObject.AddComponent<FarDistanceHeightmapGenerator>();
    }

    public void ConfigureFromGrid(OctreeGrid grid)
    {
        priority = grid != null ? grid.priority : priority;
        EnsureInitialized();
        if (grid == null) return;

        float worldSpan = Mathf.Max(512f, Mathf.Max(grid.worldSizeX, grid.worldSizeZ));
        gridWorldSize = Mathf.Max(gridWorldSize, worldSpan);

        generator.SetComputeShader(computeShaderOverride);
        generator.SetPeriodicRefresh(periodicHeightRefresh);

        generator.Configure(
            grid.worldSizeX,
            grid.worldSizeZ,
            grid.noiseSeed,
            grid.noiseTypeId,
            grid.surfaceBaseHeight,
            grid.surfaceNoiseAmplitude,
            heightTextureResolution,
            heightRefreshSeconds);
    }

    public void Tick(OctreeGrid grid)
    {
        EnsureInitialized();

        bool active = enabledSystem && priority != null;
        if (farRootObject != null && farRootObject.activeSelf != active)
            farRootObject.SetActive(active);
        if (!active) return;

        if (meshResolutionBuilt != gridResolution)
            RebuildMesh();

        Vector3 p = priority.position;
        farRootObject.transform.position = new Vector3(p.x, 0f, p.z);
        farRootObject.transform.rotation = Quaternion.identity;

        generator.Configure(
            grid.worldSizeX,
            grid.worldSizeZ,
            grid.noiseSeed,
            grid.noiseTypeId,
            grid.surfaceBaseHeight,
            grid.surfaceNoiseAmplitude,
            heightTextureResolution,
            heightRefreshSeconds);

        Texture heightTex = generator.GetOrUpdateHeightTexture(Time.time);
        if (heightTex != null)
            runtimeMaterial.SetTexture("_HeightTex", heightTex);

        runtimeMaterial.SetFloat("_UndergroundFadeMode", deepUndergroundFarFade ? 1f : 0f);
        runtimeMaterial.SetFloat("_WorldBedrockY", grid.minSubsurfaceHeight);
        runtimeMaterial.SetFloat("_PlayerHoleInner", playerHoleInnerRadius);
        runtimeMaterial.SetFloat("_PlayerHoleOuter", Mathf.Max(playerHoleInnerRadius + 1f, playerHoleOuterRadius));
        runtimeMaterial.SetFloat("_UndergroundFadeStart", undergroundFadeStartM);
        runtimeMaterial.SetFloat("_UndergroundFadeEnd", Mathf.Max(undergroundFadeStartM + 1f, undergroundFadeEndM));

        runtimeMaterial.SetVector("_WorldSizeXZ", new Vector4(grid.worldSizeX, grid.worldSizeZ, 0f, 0f));
        runtimeMaterial.SetVector("_PlayerWorldPos", new Vector4(p.x, p.y, p.z, 0f));
        runtimeMaterial.SetVector("_FarCenterWS", new Vector4(p.x, 0f, p.z, 0f));
        runtimeMaterial.SetFloat("_NearFadeStart", fadeStartDistance);
        runtimeMaterial.SetFloat("_NearFadeEnd", Mathf.Max(fadeStartDistance + 1f, fadeEndDistance));
        runtimeMaterial.SetFloat("_FarFadeStart", farFadeStartDistance);
        runtimeMaterial.SetFloat("_FarFadeEnd", Mathf.Max(farFadeStartDistance + 1f, farFadeEndDistance));
        runtimeMaterial.SetFloat("_SurfaceBaseHeight", grid.surfaceBaseHeight);
        runtimeMaterial.SetFloat("_SurfaceAmplitude", grid.surfaceNoiseAmplitude);
        runtimeMaterial.SetFloat("_FogHorizonMix", fogHorizonMix);
        runtimeMaterial.SetFloat("_UnderlayYOffset", underlayYOffset);
        runtimeMaterial.SetColor("_FarColor", farColor);
        runtimeMaterial.SetColor("_HorizonColor", horizonFogColor);
        runtimeMaterial.SetFloat("_BackgroundClipEpsilon", Mathf.Max(1e-7f, backgroundClipEpsilon));

        if (farMesh != null)
            farMesh.bounds = new Bounds(Vector3.zero, new Vector3(HugeBoundsExtent, HugeBoundsExtent, HugeBoundsExtent));
    }

    private void CreateFarMeshObject()
    {
        farRootObject = new GameObject("FarDistanceRoot");
        farRootObject.transform.SetParent(transform, false);

        farMeshObject = new GameObject("FarDistanceHeightmap");
        farMeshObject.transform.SetParent(farRootObject.transform, false);

        int layer = priority != null ? priority.gameObject.layer : gameObject.layer;
        farRootObject.layer = layer;
        farMeshObject.layer = layer;
        farMeshFilter = farMeshObject.AddComponent<MeshFilter>();
        farMeshRenderer = farMeshObject.AddComponent<MeshRenderer>();

        Shader s = Shader.Find("Voxel Void/FarDistanceHeightmap");
        if (s == null)
            s = Shader.Find("Universal Render Pipeline/Unlit");
        runtimeMaterial = materialOverride != null
            ? new Material(materialOverride)
            : new Material(s);
        runtimeMaterial.name = "FarDistanceHeightmap_RuntimeMaterial";
        runtimeMaterial.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Geometry - 50;

        farMeshRenderer.sharedMaterial = runtimeMaterial;
        farMeshRenderer.allowOcclusionWhenDynamic = false;
        farMeshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        farMeshRenderer.receiveShadows = false;
        RebuildMesh();
    }

    private void RebuildMesh()
    {
        if (farMesh != null)
            Destroy(farMesh);

        farMesh = BuildGridMesh(gridResolution, gridWorldSize);
        farMesh.name = "FarDistanceHeightmap_Mesh";
        farMeshFilter.sharedMesh = farMesh;
        meshResolutionBuilt = gridResolution;
        if (farMesh != null)
            farMesh.bounds = new Bounds(Vector3.zero, new Vector3(HugeBoundsExtent, HugeBoundsExtent, HugeBoundsExtent));
    }

    private static Mesh BuildGridMesh(int resolution, float size)
    {
        int vertsPerAxis = resolution + 1;
        int vertCount = vertsPerAxis * vertsPerAxis;
        int triCount = resolution * resolution * 2;

        Vector3[] vertices = new Vector3[vertCount];
        Vector2[] uvs = new Vector2[vertCount];
        int[] triangles = new int[triCount * 3];

        float half = size * 0.5f;
        int vi = 0;
        for (int z = 0; z <= resolution; z++)
        {
            float tz = (float)z / resolution;
            float pz = Mathf.Lerp(-half, half, tz);
            for (int x = 0; x <= resolution; x++)
            {
                float tx = (float)x / resolution;
                float px = Mathf.Lerp(-half, half, tx);
                vertices[vi] = new Vector3(px, 0f, pz);
                uvs[vi] = new Vector2(tx, tz);
                vi++;
            }
        }

        int ti = 0;
        for (int z = 0; z < resolution; z++)
        {
            int row = z * vertsPerAxis;
            int nextRow = (z + 1) * vertsPerAxis;
            for (int x = 0; x < resolution; x++)
            {
                int a = row + x;
                int b = a + 1;
                int c = nextRow + x;
                int d = c + 1;

                triangles[ti++] = a;
                triangles[ti++] = c;
                triangles[ti++] = b;

                triangles[ti++] = b;
                triangles[ti++] = c;
                triangles[ti++] = d;
            }
        }

        var mesh = new Mesh();
        if (vertCount > 65000)
            mesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;
        mesh.vertices = vertices;
        mesh.uv = uvs;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();
        return mesh;
    }
}
