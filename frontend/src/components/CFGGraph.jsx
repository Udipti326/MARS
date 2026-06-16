import { useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";

function colorForNode(node) {
  if (node.type === "expedition" && node.role === "current") return "#ffffff";
  if (node.type === "expedition" && node.role === "related") return "#60a5fa";
  if (node.type === "claim") return "#8c8c8c";
  return "#cfcfcf";
}

function radiusForNode(node) {
  if (node.type === "expedition") return node.role === "current" ? 12 : 10;
  if (node.type === "claim") return 7;
  return 6;
}

function CFGGraph({ data, onNodeClick, selectedNodeId }) {
  const graphData = useMemo(() => {
    const nodes = Array.isArray(data?.nodes) ? data.nodes : [];
    const nodeIds = new Set(nodes.map((n) => String(n.id)));

    const links = Array.isArray(data?.links)
      ? data.links
          .filter(
            (l) =>
              nodeIds.has(String(l.source)) &&
              nodeIds.has(String(l.target))
          )
          .map((l) => ({
            ...l,
            source: String(l.source),
            target: String(l.target),
          }))
      : [];

    return { nodes, links };
  }, [data]);

  if (!graphData.nodes.length) {
    return <div className="empty">No CFG graph available yet.</div>;
  }

  return (
    <div
      style={{
        width: "100%",
        height: "640px",
        border: "1px solid #2a2a2a",
        borderRadius: "18px",
        overflow: "hidden",
      }}
    >
      <ForceGraph2D
        graphData={graphData}
        nodeId="id"
        backgroundColor="#111111"
        nodeLabel={(node) => {
          const label = node?.label || node?.id || "Node";
          const semantic = Number(
            node?.relevance_score ??
              node?.similarity_to_current ??
              node?.score ??
              node?.confidence ??
              0
          ).toFixed(6);
          const parent = node?.parent_expedition_title
            ? `\nParent: ${node.parent_expedition_title}`
            : "";
          const role = node?.role ? `\nRole: ${node.role}` : "";
          return `${label}\nType: ${node?.type || "unknown"}${role}${parent}\nSemantic score: ${semantic}`;
        }}
        linkLabel={(link) =>
          `${link.type || "related"}: ${Number(link.weight || 0).toFixed(6)}`
        }
        nodeCanvasObject={(node, ctx, globalScale) => {
          const x = node.x || 0;
          const y = node.y || 0;
          const radius = radiusForNode(node);

          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
          ctx.fillStyle =
            String(node.id) === String(selectedNodeId)
              ? "#f59e0b"
              : colorForNode(node);
          ctx.fill();

          if (String(node.id) === String(selectedNodeId)) {
            ctx.lineWidth = Math.max(1, 2 / globalScale);
            ctx.strokeStyle = "#f59e0b";
            ctx.stroke();
          }

          // Only expedition parent nodes are visible on the canvas.
          if (node.type === "expedition") {
            const label = node.label || node.id;
            const fontSize = Math.max(10, 14 / globalScale);
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#f3f3f3";
            ctx.fillText(label, x, y + 16);
          }
        }}
        nodePointerAreaPaint={(node, color, ctx) => {
          const x = node.x || 0;
          const y = node.y || 0;
          const radius = radiusForNode(node);

          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x, y, radius, 0, 2 * Math.PI, false);
          ctx.fill();
        }}
        linkWidth={(link) => 1 + Math.max(0, Number(link.weight || 0)) * 6}
        linkColor={(link) => {
          if (link.type === "RELATED_RESEARCH") return "#3b82f6";
          if (link.type === "RELATED_TOPIC") return "#7c3aed";
          if (link.type === "CLAIM_TO_TOPIC") return "#f59e0b";
          return "#666";
        }}
        onNodeClick={(node) => onNodeClick?.(node)}
      />
    </div>
  );
}

export default CFGGraph;