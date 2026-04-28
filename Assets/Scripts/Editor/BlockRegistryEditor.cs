#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

/// <summary>
/// Custom inspector for BlockRegistry.
/// Adds a "Rebuild Registry" button and shows a summary of registered blocks.
/// Place this file inside any Editor/ folder in your project.
/// </summary>
[CustomEditor(typeof(BlockRegistry))]
public class BlockRegistryEditor : Editor
{
    public override void OnInspectorGUI()
    {
        var registry = (BlockRegistry)target;

        DrawDefaultInspector();

        EditorGUILayout.Space(8);

        // ── Status ────────────────────────────────────────────────────────
        if (registry.textureArray != null)
        {
            EditorGUILayout.HelpBox(
                $"Texture array ready — {registry.textureArray.depth} layers " +
                $"at {registry.textureArray.width}×{registry.textureArray.height}." +
                $"Drag 'Texture Array' field to your voxel material _MainTex slot.",
                MessageType.Info);
        }
        else
        {
            EditorGUILayout.HelpBox(
                "No texture array yet. Add blocks and click Rebuild.",
                MessageType.Warning);
        }

        EditorGUILayout.Space(8);
        EditorGUILayout.LabelField("Blocks registered", EditorStyles.boldLabel);

        if (registry.blocks != null)
        {
            foreach (var block in registry.blocks)
            {
                if (block == null) continue;
                EditorGUILayout.LabelField(
                    $"  [{block.blockId:D3}]  {block.blockName}",
                    EditorStyles.miniLabel);
            }
        }

        EditorGUILayout.Space(8);

        if (GUILayout.Button("Rebuild Registry  (textures + lookup)", GUILayout.Height(32)))
        {
            registry.Rebuild();
            AssetDatabase.SaveAssets();
            Debug.Log("[BlockRegistryEditor] Registry rebuilt successfully.");
        }
    }
}
#endif