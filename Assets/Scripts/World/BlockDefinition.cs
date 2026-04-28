using UnityEngine;

/// <summary>
/// One ScriptableObject asset per block type.
///
/// SETUP
/// ─────
/// Assets > Create > Voxel Void > Block Definition
///
/// Fill in the fields, assign textures, set a unique blockId (1–254).
/// blockId 0 is reserved for Air — never create a block with id 0.
///
/// TEXTURE LAYOUT
/// ──────────────
/// The engine uses a Texture2DArray (built by BlockRegistry) so all blocks
/// share a single draw call. Each face can point to a different texture slot:
///
///   topTexture    — face pointing up   (grass top, snow, etc.)
///   sideTexture   — faces pointing N/S/E/W
///   bottomTexture — face pointing down (dirt under grass, etc.)
///
/// If sideTexture and bottomTexture are left null they fall back to topTexture,
/// so a uniform block only needs one texture assigned.
///
/// ADDING A NEW BLOCK
/// ──────────────────
/// 1. Create a new BlockDefinition asset.
/// 2. Assign a unique blockId.
/// 3. Assign textures (PNG/TGA, power-of-two resolution, same size as others).
/// 4. Open the BlockRegistry asset and add this definition to the list.
/// 5. Press "Rebuild Registry" in the BlockRegistry inspector — done.
/// </summary>
[CreateAssetMenu(menuName = "Voxel Void/Block Definition", fileName = "Block_New")]
public class BlockDefinition : ScriptableObject
{
    [Header("Identity")]
    [Tooltip("Unique id used throughout the engine. 0 = Air (reserved). Range 1–254.")]
    [Range(1, 254)]
    public byte blockId = 1;

    [Tooltip("Human-readable name shown in the editor and debug UI.")]
    public string blockName = "New Block";

    [Header("Textures")]
    [Tooltip("Texture for the top face (+Y). Used for all faces if side/bottom are null.")]
    public Texture2D topTexture;

    [Tooltip("Texture for the four side faces (N/S/E/W). Falls back to topTexture if null.")]
    public Texture2D sideTexture;

    [Tooltip("Texture for the bottom face (-Y). Falls back to topTexture if null.")]
    public Texture2D bottomTexture;

    [Header("Physics")]
    [Tooltip("Whether this block has a MeshCollider. Decorative blocks (leaves, glass) may skip collision.")]
    public bool hasCollision = true;

    [Header("Rendering")]
    [Tooltip("If true, neighbouring solid faces are still emitted (for transparent blocks like glass/water).")]
    public bool isTransparent = false;

    // ------------------------------------------------------------------
    //  Helpers
    // ------------------------------------------------------------------

    /// <summary>
    /// Returns the correct texture for a given face side index.
    /// Side indices match Tables.Normals: 0=+Y(top) 1=-Y(bottom) 2–5=sides.
    /// </summary>
    public Texture2D GetTextureForSide(int side)
    {
        switch (side)
        {
            case 0: return topTexture;
            case 1: return bottomTexture != null ? bottomTexture : topTexture;
            default: return sideTexture  != null ? sideTexture  : topTexture;
        }
    }

    /// <summary>
    /// Texture array layer index for a given face side.
    /// Set by BlockRegistry.Rebuild() — do not set manually.
    /// Index layout: blockId * 3 + faceSlot (0=top, 1=bottom, 2=side)
    /// </summary>
    [HideInInspector] public int texArrayLayerTop    = 0;
    [HideInInspector] public int texArrayLayerBottom = 0;
    [HideInInspector] public int texArrayLayerSide   = 0;

    /// <summary>Returns the tex array layer for a given side index (0=+Y … 5=side).</summary>
    public int GetTexLayerForSide(int side)
    {
        switch (side)
        {
            case 0: return texArrayLayerTop;
            case 1: return texArrayLayerBottom;
            default: return texArrayLayerSide;
        }
    }
}
