using UnityEngine;

/// <summary>
/// Handles player input for breaking and placing voxel blocks.
///
/// =========================================================================
/// SETUP — siga estes passos no Unity Editor:
/// =========================================================================
///
/// 1. ADICIONAR O SCRIPT
///    - Selecione o GameObject do player na Hierarchy.
///    - Add Component → BlockInteraction.
///
/// 2. PREENCHER OS CAMPOS NO INSPECTOR
///    - playerCamera : arraste a Camera que o player usa (geralmente "Main Camera").
///    - octreeGrid   : arraste o GameObject que contém o script OctreeGrid.
///                     !! NÃO use mais o campo "octree" (Octree individual) !!
///    - reach        : distância máxima de alcance (padrão: 10 unidades).
///    - terrainLayer : deixe como "Everything" para começar.
///
/// 3. WORLDMODIFICATIONS NA CENA
///    - Crie um GameObject vazio → renomeie para "WorldModifications".
///    - Add Component → WorldModifications.
///
/// 4. MESHCOLLIDER (feito automaticamente pelo Node.cs)
///    - Node.cs já adiciona MeshCollider em cada chunk. Nada a fazer aqui.
///
/// 5. CONTROLES
///    - Botão esquerdo do mouse  = quebrar bloco
///    - Botão direito do mouse   = colocar bloco
///
/// =========================================================================
/// COMO FUNCIONA INTERNAMENTE:
/// =========================================================================
///
/// 1. Todo frame, Physics.Raycast é disparado do centro da tela.
/// 2. No hit, calculamos qual voxel foi atingido usando a normal da superfície.
/// 3. A posição é registrada em WorldModifications e o nó é localizado via
///    OctreeGrid.FindLeafAt() — que itera por todos os Octrees activos.
/// 4. O nó e seus 6 vizinhos (que podem estar em Octrees diferentes) são
///    marcados como dirty e forçados a re-mesh imediatamente no mesmo frame,
///    usando um mini flush de jobs — faces vizinhas sempre aparecem juntas.
/// =========================================================================
/// </summary>
public class BlockInteraction : MonoBehaviour
{
    [Header("Referências")]
    [Tooltip("A câmera principal do player.")]
    public Camera playerCamera;

    [Tooltip("O GameObject que contém o script OctreeGrid (não mais um Octree individual).")]
    public OctreeGrid octreeGrid;

    [Header("Configurações")]
    [Tooltip("Distância máxima de alcance para quebrar/colocar blocos.")]
    public float reach = 10f;

    [Tooltip("Layers que contam como terreno.")]
    public LayerMask terrainLayer = ~0;

    // Nudge dentro/fora da face para acertar o voxel correto
    private const float NudgeInto = 0.01f;
    private const float NudgeOut = 0.99f;

    // -----------------------------------------------------------------------

    private void Update()
    {
        if (Input.GetMouseButtonDown(0)) TryBreak();
        if (Input.GetMouseButtonDown(1)) TryPlace();
    }

    private void OnGUI()
    {
        float cx = Screen.width * 0.5f;
        float cy = Screen.height * 0.5f;
        float s = 10f;
        GUI.color = Color.white;
        GUI.DrawTexture(new Rect(cx - 1, cy - s, 2, s * 2), Texture2D.whiteTexture);
        GUI.DrawTexture(new Rect(cx - s, cy - 1, s * 2, 2), Texture2D.whiteTexture);

        GUILayout.Space(80);
        GUI.color = Color.white;
        GUILayout.Label("LMB = quebrar bloco   RMB = colocar bloco");
        if (WorldModifications.Instance != null)
            GUILayout.Label($"Modificações armazenadas: {WorldModifications.Instance.Count}");
    }

    // -----------------------------------------------------------------------
    //  Ações
    // -----------------------------------------------------------------------

    private void TryBreak()
    {
        if (!Raycast(out RaycastHit hit)) return;

        // Entra levemente na face → voxel sólido atingido
        Vector3 voxelPos = hit.point - hit.normal * NudgeInto;
        WorldModifications.Instance.Set(voxelPos, WorldModifications.VoxelState.Air);
        RebuildNodeAndNeighboursAt(voxelPos);
    }

    private void TryPlace()
    {
        if (!Raycast(out RaycastHit hit)) return;

        // Sai levemente da face → célula de ar adjacente
        Vector3 voxelPos = hit.point + hit.normal * NudgeOut;
        WorldModifications.Instance.Set(voxelPos, WorldModifications.VoxelState.Solid);
        RebuildNodeAndNeighboursAt(voxelPos);
    }

    // -----------------------------------------------------------------------
    //  Rebuild
    // -----------------------------------------------------------------------

    private void RebuildNodeAndNeighboursAt(Vector3 worldPos)
    {
        // Validate setup
        if (octreeGrid == null)
        {
            Debug.LogError(
                "[BlockInteraction] OctreeGrid não está atribuído!\n" +
                "Arraste o GameObject com o script OctreeGrid para o campo 'Octree Grid' " +
                "no Inspector do BlockInteraction.");
            return;
        }

        if (WorldModifications.Instance == null)
        {
            Debug.LogError(
                "[BlockInteraction] WorldModifications não encontrado na cena!\n" +
                "Crie um GameObject vazio e adicione o script WorldModifications nele.");
            return;
        }

        // Locate the leaf node that contains this voxel
        Node node = octreeGrid.FindLeafAt(worldPos);
        if (node == null)
        {
            Debug.LogWarning($"[BlockInteraction] Nenhum nó encontrado em {worldPos}");
            return;
        }

        // Collect all nodes that need rebuilding (target + 6 face neighbours)
        // Using a small fixed-size list avoids allocation pressure.
        Node[] nodesToRebuild = new Node[7];
        int count = 0;

        nodesToRebuild[count++] = node;

        // The neighbour search step equals the leaf's world size so we land
        // squarely inside the adjacent chunk, regardless of which OctreeGrid
        // cell it belongs to.
        float step = node.NodeScale();
        Vector3 center = node.NodePosition();

        Vector3[] offsets =
        {
            Vector3.right   * step,
            Vector3.left    * step,
            Vector3.up      * step,
            Vector3.down    * step,
            Vector3.forward * step,
            Vector3.back    * step,
        };

        foreach (Vector3 o in offsets)
        {
            // FindLeafAt searches across ALL active OctreeGrid cells
            Node neighbour = octreeGrid.FindLeafAt(center + o);
            if (neighbour != null && neighbour != node)
                nodesToRebuild[count++] = neighbour;
        }

        // Mark all dirty first, then flush them together in one mini-batch
        // so every face is updated in the same frame (no one-frame flicker).
        for (int i = 0; i < count; i++)
            nodesToRebuild[i].MarkDirty();

        FlushNodes(nodesToRebuild, count);
    }

    /// <summary>
    /// Collects, schedules, completes and applies mesh jobs for the given
    /// nodes immediately — within the same frame as the player's click.
    /// This mirrors what OctreeGrid.LateUpdate does but only for these nodes.
    /// </summary>
    private void FlushNodes(Node[] nodes, int count)
    {
        // Collect completers
        var completers = new JobCompleter[count];
        int jobCount = 0;

        for (int i = 0; i < count; i++)
        {
            if (nodes[i].TryCollectJob(out JobCompleter jc))
                completers[jobCount++] = jc;
        }

        if (jobCount == 0) return;

        // Schedule all in one batch (maximises Burst parallelism)
        var handles = new Unity.Collections.NativeArray<Unity.Jobs.JobHandle>(
            jobCount, Unity.Collections.Allocator.Temp);

        for (int i = 0; i < jobCount; i++)
            handles[i] = completers[i].schedule();

        Unity.Jobs.JobHandle.CompleteAll(handles);
        handles.Dispose();

        // Apply finished meshes on the main thread
        for (int i = 0; i < jobCount; i++)
            completers[i].onComplete();
    }

    // -----------------------------------------------------------------------

    private bool Raycast(out RaycastHit hit)
    {
        if (WorldModifications.Instance == null)
        {
            Debug.LogError(
                "[BlockInteraction] WorldModifications não encontrado na cena!\n" +
                "Crie um GameObject vazio e adicione o script WorldModifications nele.");
            hit = default;
            return false;
        }

        Ray ray = playerCamera.ScreenPointToRay(
            new Vector3(Screen.width * 0.5f, Screen.height * 0.5f, 0f));

        return Physics.Raycast(ray, out hit, reach, terrainLayer);
    }
}