using UnityEngine;

/// <summary>
/// Builds a tileable runtime height texture aligned with the world's toroidal period.
/// </summary>
public class FarDistanceHeightmapGenerator : MonoBehaviour
{
    [SerializeField] private bool periodicRefresh = false;
    private float worldSizeX = 50000f;
    private float worldSizeZ = 50000f;
    private int noiseSeed = 2376;
    private int noiseTypeId = 0;
    private float surfaceBaseHeight = 0f;
    private float surfaceNoiseAmplitude = 300f;
    private int resolution = 1024;
    private float refreshSeconds = 0.75f;

    [SerializeField] private ComputeShader computeShader;

    private Texture2D heightTexCpu;
    private RenderTexture heightTexGpu;
    private Color[] pixels;
    private float nextRefreshTime = float.NegativeInfinity;
    private bool dirty = true;

    private FastNoiseLite _pointSampleNoise;
    private bool _pointSampleNoiseCreated;
    private int _pointSampleNoiseVersion = int.MinValue;

    public void Configure(
        float worldX,
        float worldZ,
        int seed,
        int noiseType,
        float baseHeight,
        float amplitude,
        int texResolution,
        float refreshInterval)
    {
        bool changed =
            !Mathf.Approximately(worldSizeX, worldX) ||
            !Mathf.Approximately(worldSizeZ, worldZ) ||
            noiseSeed != seed ||
            noiseTypeId != noiseType ||
            !Mathf.Approximately(surfaceBaseHeight, baseHeight) ||
            !Mathf.Approximately(surfaceNoiseAmplitude, amplitude) ||
            resolution != texResolution ||
            !Mathf.Approximately(refreshSeconds, refreshInterval);

        worldSizeX = Mathf.Max(1f, worldX);
        worldSizeZ = Mathf.Max(1f, worldZ);
        noiseSeed = seed;
        noiseTypeId = noiseType;
        surfaceBaseHeight = baseHeight;
        surfaceNoiseAmplitude = Mathf.Max(0.001f, Mathf.Abs(amplitude));
        resolution = Mathf.Clamp(texResolution, 128, 2048);
        refreshSeconds = Mathf.Max(0.02f, refreshInterval);

        if (changed)
        {
            dirty = true;
            _pointSampleNoiseVersion = int.MinValue;
        }
    }

    /// <summary>
    /// Same height field as the runtime texture (CPU noise path), for a single world XZ sample per frame.
    /// </summary>
    public float SampleSurfaceHeightWorld(float wxWorld, float wzWorld)
    {
        if (!_pointSampleNoiseCreated)
        {
            _pointSampleNoise = new FastNoiseLite();
            _pointSampleNoiseCreated = true;
        }

        int ver = noiseSeed ^ (noiseTypeId * 131) ^ Mathf.RoundToInt(worldSizeX) ^ (Mathf.RoundToInt(worldSizeZ) * 17)
            ^ Mathf.RoundToInt(surfaceBaseHeight * 10f) ^ Mathf.RoundToInt(surfaceNoiseAmplitude * 10f);
        if (ver != _pointSampleNoiseVersion)
        {
            _pointSampleNoiseVersion = ver;
            _pointSampleNoise.SetSeed(noiseSeed);
            _pointSampleNoise.SetFrequency(0.003f);
            switch (noiseTypeId)
            {
                case 1: _pointSampleNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2S); break;
                case 2: _pointSampleNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin); break;
                default: _pointSampleNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2); break;
            }
        }

        return SampleSurfaceHeight(wxWorld, wzWorld, ref _pointSampleNoise);
    }

    public void SetComputeShader(ComputeShader shader)
    {
        computeShader = shader;
    }

    public void SetPeriodicRefresh(bool enabled)
    {
        periodicRefresh = enabled;
    }

    public Texture GetOrUpdateHeightTexture(float timeNow)
    {
        if (!dirty && (!periodicRefresh || timeNow < nextRefreshTime))
            return heightTexGpu != null ? (Texture)heightTexGpu : heightTexCpu;

        if (SystemInfo.supportsComputeShaders && computeShader != null)
        {
            RecreateGpuTextureIfNeeded();
            BuildHeightTextureGpu();
        }
        else
        {
            RecreateCpuTextureIfNeeded();
            BuildHeightTextureCpu();
        }

        dirty = false;
        nextRefreshTime = timeNow + refreshSeconds;
        return heightTexGpu != null ? (Texture)heightTexGpu : heightTexCpu;
    }

    private void RecreateCpuTextureIfNeeded()
    {
        if (heightTexCpu != null && heightTexCpu.width == resolution && heightTexCpu.height == resolution)
            return;

        if (heightTexCpu != null)
            Destroy(heightTexCpu);

        heightTexCpu = new Texture2D(resolution, resolution, TextureFormat.RHalf, false, true)
        {
            name = "FarDistanceHeightRuntime",
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Bilinear
        };
        pixels = new Color[resolution * resolution];
        dirty = true;
    }

    private void RecreateGpuTextureIfNeeded()
    {
        if (heightTexGpu != null && heightTexGpu.width == resolution && heightTexGpu.height == resolution)
            return;

        if (heightTexGpu != null)
            heightTexGpu.Release();

        heightTexGpu = new RenderTexture(resolution, resolution, 0, RenderTextureFormat.RHalf)
        {
            name = "FarDistanceHeightRuntimeRT",
            enableRandomWrite = true,
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Bilinear
        };
        heightTexGpu.Create();
    }

    private void BuildHeightTextureGpu()
    {
        int kernel = computeShader.FindKernel("CSMain");
        computeShader.SetTexture(kernel, "Result", heightTexGpu);
        computeShader.SetInt("_Resolution", resolution);
        computeShader.SetFloat("_WorldSizeX", worldSizeX);
        computeShader.SetFloat("_WorldSizeZ", worldSizeZ);
        computeShader.SetInt("_NoiseSeed", noiseSeed);
        computeShader.SetInt("_NoiseTypeId", noiseTypeId);
        computeShader.SetFloat("_SurfaceBaseHeight", surfaceBaseHeight);
        computeShader.SetFloat("_SurfaceAmplitude", surfaceNoiseAmplitude);

        int groups = Mathf.CeilToInt(resolution / 8f);
        computeShader.Dispatch(kernel, groups, groups, 1);
    }

    private void BuildHeightTextureCpu()
    {
        FastNoiseLite surfaceNoise = new FastNoiseLite();
        surfaceNoise.SetSeed(noiseSeed);
        surfaceNoise.SetFrequency(0.003f);
        switch (noiseTypeId)
        {
            case 1: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2S); break;
            case 2: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.Perlin); break;
            default: surfaceNoise.SetNoiseType(FastNoiseLite.NoiseType.OpenSimplex2); break;
        }

        int idx = 0;
        for (int z = 0; z < resolution; z++)
        {
            float vz = (float)z / (resolution - 1);
            for (int x = 0; x < resolution; x++)
            {
                float vx = (float)x / (resolution - 1);

                float wx = vx * worldSizeX;
                float wz = vz * worldSizeZ;

                float h = SampleSurfaceHeight(wx, wz, ref surfaceNoise);
                float normalized = Mathf.Clamp01((h - (surfaceBaseHeight - surfaceNoiseAmplitude)) / (2f * surfaceNoiseAmplitude));
                pixels[idx++] = new Color(normalized, 0f, 0f, 1f);
            }
        }

        heightTexCpu.SetPixels(pixels);
        heightTexCpu.Apply(false, false);
    }

    private float SampleSurfaceHeight(float wxWorld, float wzWorld, ref FastNoiseLite surfaceNoise)
    {
        ComputeToroidalNoiseCoords(wxWorld, wzWorld, out float cx, out float sx, out float cz, out float sz);

        float n1 = surfaceNoise.GetNoise(cx, sx, cz);
        float n2 = surfaceNoise.GetNoise(cz + 100f, sz + 100f, cx * 0.5f);
        float n3 = surfaceNoise.GetNoise(sx * 0.7f, sz * 0.7f + 200f, cz * 0.3f);

        return surfaceBaseHeight + (n1 * 0.6f + n2 * 0.3f + n3 * 0.1f) * surfaceNoiseAmplitude;
    }

    private void ComputeToroidalNoiseCoords(float wxWorld, float wzWorld,
        out float cx, out float sx, out float cz, out float sz)
    {
        float wxLin = PosMod(wxWorld, worldSizeX);
        float wzLin = PosMod(wzWorld, worldSizeZ);

        float aX = wxLin / worldSizeX * 2f * Mathf.PI;
        float aZ = wzLin / worldSizeZ * 2f * Mathf.PI;
        float r = worldSizeX / (2f * Mathf.PI);
        float s = worldSizeZ / (2f * Mathf.PI);

        cx = Mathf.Cos(aX) * r;
        sx = Mathf.Sin(aX) * r;
        cz = Mathf.Cos(aZ) * s;
        sz = Mathf.Sin(aZ) * s;
    }

    private static float PosMod(float x, float m)
    {
        float r = x % m;
        return r < 0f ? r + m : r;
    }
}
