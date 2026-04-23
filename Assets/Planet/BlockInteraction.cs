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
///    - octree       : arraste o GameObject que contém o script Octree.
///    - reach        : distância máxima de alcance (padrão: 10 unidades).
///    - terrainLayer : deixe como "Everything" para começar; restrinja depois
///                     se necessário.
///
/// 3. WORLDMODIFICATIONS NA CENA
///    - Crie um GameObject vazio: GameObject → Create Empty → renomeie para
///      "WorldModifications".
///    - Add Component → WorldModifications.
///    - Este objeto guarda todas as modificações de blocos em memória.
///
/// 4. MESHCOLLIDER (feito automaticamente pelo Node.cs)
///    - Node.cs já adiciona MeshCollider automaticamente em cada chunk.
///    - Você NÃO precisa adicionar MeshCollider manualmente no prefab.
///    - Se o prefab já tiver um MeshCollider, ele será reutilizado.
///
/// 5. CONTROLES
///    - Botão esquerdo do mouse  = quebrar bloco
///    - Botão direito do mouse   = colocar bloco
///    - Um crosshair simples é desenhado no centro da tela via OnGUI.
///
/// =========================================================================
/// COMO FUNCIONA INTERNAMENTE:
/// =========================================================================
///
/// 1. Todo frame, Physics.Raycast é disparado do centro da tela.
/// 2. No hit, calculamos qual voxel foi atingido usando a normal da superfície:
///      - Quebrar: passo levemente PARA DENTRO da face atingida (NudgeInto).
///      - Colocar: passo levemente PARA FORA da face atingida (NudgeOut).
/// 3. A posição do voxel é registrada em WorldModifications (singleton).
///    A chave usada é a posição WRAPEADA — então um bloco em X=4096 e X=0
///    têm a mesma chave e são a mesma modificação. Isso é o que permite
///    "voltar ao ponto de origem" e ver o que foi construído/destruído.
/// 4. O nó folha que contém o voxel é localizado via Octree.FindLeafAt()
///    e marcado como dirty (MarkDirty), forçando re-mesh no próximo frame.
/// 5. Os 6 vizinhos do nó também são marcados para corrigir as faces
///    compartilhadas na borda entre chunks.
/// =========================================================================
/// </summary>
public class BlockInteraction : MonoBehaviour
{
    [Header("Referências")]
    [Tooltip("A câmera principal do player (usada para o raycast).")]
    public Camera playerCamera;

    [Tooltip("O GameObject que contém o script Octree.")]
    public Octree octree;

    [Header("Configurações")]
    [Tooltip("Distância máxima de alcance para quebrar/colocar blocos.")]
    public float reach = 10f;

    [Tooltip("Layers que contam como terreno. 'Everything' funciona para começar.")]
    public LayerMask terrainLayer = ~0;

    // Quanto avançar dentro/fora da face para acertar o voxel correto
    private const float NudgeInto = 0.01f;  // quebrar: entra levemente na face
    private const float NudgeOut = 0.99f;  // colocar: sai levemente da face

    // -----------------------------------------------------------------------

    private void Update()
    {
        if (Input.GetMouseButtonDown(0)) TryBreak();
        if (Input.GetMouseButtonDown(1)) TryPlace();
    }

    private void OnGUI()
    {
        // Crosshair simples no centro da tela
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

        // Avança levemente PARA DENTRO da face para cair no voxel sólido atingido
        Vector3 voxelPos = hit.point - hit.normal * NudgeInto;
        WorldModifications.Instance.Set(voxelPos, WorldModifications.VoxelState.Air);
        RebuildNodeAt(voxelPos);
    }

    private void TryPlace()
    {
        if (!Raycast(out RaycastHit hit)) return;

        // Avança levemente PARA FORA da face para cair na célula de ar adjacente
        Vector3 voxelPos = hit.point + hit.normal * NudgeOut;
        WorldModifications.Instance.Set(voxelPos, WorldModifications.VoxelState.Solid);
        RebuildNodeAt(voxelPos);
    }

    // -----------------------------------------------------------------------
    //  Rebuild
    // -----------------------------------------------------------------------

    private void RebuildNodeAt(Vector3 worldPos)
    {
        Node node = octree.FindLeafAt(worldPos);
        if (node == null)
        {
            UnityEngine.Debug.LogWarning($"[BlockInteraction] Nenhum nó encontrado em {worldPos}");
            return;
        }

        node.MarkDirty();
        RebuildNeighboursOf(node);
    }

    /// <summary>
    /// Marca os 6 vizinhos do nó para re-mesh, corrigindo as faces
    /// compartilhadas na borda entre chunks adjacentes.
    /// </summary>
    private void RebuildNeighboursOf(Node node)
    {
        float step = node.NodeScale();
        Vector3 center = node.NodePosition();

        Vector3[] neighbourOffsets =
        {
            Vector3.right   * step,
            Vector3.left    * step,
            Vector3.up      * step,
            Vector3.down    * step,
            Vector3.forward * step,
            Vector3.back    * step,
        };

        foreach (Vector3 o in neighbourOffsets)
        {
            Node neighbour = octree.FindLeafAt(center + o);
            if (neighbour != null && neighbour != node)
                neighbour.MarkDirty();
        }
    }

    // -----------------------------------------------------------------------

    private bool Raycast(out RaycastHit hit)
    {
        // Verifica se WorldModifications está na cena
        if (WorldModifications.Instance == null)
        {
            UnityEngine.Debug.LogError(
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