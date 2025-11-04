"""Scenario builders for SUMO traffic simulations."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class ScenarioArtifacts:
    """Handles references to generated SUMO files for a scenario."""

    directory: Path
    net_file: Path
    route_file: Path
    config_file: Path
    tls_id: str
    incoming_lanes: List[str] = field(default_factory=list)
    outgoing_lanes: List[str] = field(default_factory=list)
    plain_files: List[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        """Delete generated scenario files."""

        if self.directory.exists():
            shutil.rmtree(self.directory, ignore_errors=True)


class SimpleIntersectionScenario:
    """Generate a minimal four-way intersection controlled by a traffic light."""

    INCOMING_EDGES = ("north_in", "south_in", "east_in", "west_in")
    OUTGOING_EDGES = ("south_out", "north_out", "west_out", "east_out")

    def __init__(
        self,
        lane_length: float = 100.0,
        flow_rate: int = 600,
        output_dir: Optional[Path] = None,
        tls_id: str = "J0",
    ) -> None:
        self.lane_length = lane_length
        self.flow_rate = flow_rate
        self.tls_id = tls_id
        self._provided_dir = output_dir
        self._artifacts: Optional[ScenarioArtifacts] = None

    def build(self, regenerate: bool = False) -> ScenarioArtifacts:
        """Create the SUMO network, routes, and configuration files."""

        if self._artifacts is not None and not regenerate:
            return self._artifacts

        temp_dir = Path(self._provided_dir) if self._provided_dir else Path(
            tempfile.mkdtemp(prefix="sumo-sim-")
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        net_file = temp_dir / "network.net.xml"
        route_file = temp_dir / "routes.rou.xml"
        config_file = temp_dir / "config.sumocfg"

        incoming_lanes, outgoing_lanes, plain_files = self._generate_network(temp_dir, net_file)
        self._write_route_file(route_file)
        self._write_config_file(config_file, net_file, route_file)

        self._artifacts = ScenarioArtifacts(
            directory=temp_dir,
            net_file=net_file,
            route_file=route_file,
            config_file=config_file,
            tls_id=self.tls_id,
            incoming_lanes=incoming_lanes,
            outgoing_lanes=outgoing_lanes,
            plain_files=plain_files,
        )
        return self._artifacts

    # ------------------------------------------------------------------
    # Network generation helpers
    # ------------------------------------------------------------------
    def _generate_network(self, temp_dir: Path, net_file: Path) -> Tuple[List[str], List[str], List[Path]]:
        nodes_file = temp_dir / "nodes.nod.xml"
        edges_file = temp_dir / "edges.edg.xml"
        connections_file = temp_dir / "connections.con.xml"

        self._write_nodes_file(nodes_file)
        self._write_edges_file(edges_file)
        self._write_connections_file(connections_file)

        cmd = [
            self._resolve_netconvert_binary(),
            "--node-files",
            os.fspath(nodes_file),
            "--edge-files",
            os.fspath(edges_file),
            "--connection-files",
            os.fspath(connections_file),
            "--output-file",
            os.fspath(net_file),
            "--no-internal-links",
        ]

        subprocess.run(cmd, check=True)

        incoming_lanes, outgoing_lanes = self._extract_lanes_and_update_tl(net_file)

        return incoming_lanes, outgoing_lanes, [nodes_file, edges_file, connections_file]

    def _write_nodes_file(self, nodes_file: Path) -> None:
        root = ET.Element("nodes")
        length = self.lane_length

        for node_id, x_pos, y_pos, node_type in (
            ("north", 0.0, length, "priority"),
            ("south", 0.0, -length, "priority"),
            ("east", length, 0.0, "priority"),
            ("west", -length, 0.0, "priority"),
            (self.tls_id, 0.0, 0.0, "traffic_light"),
        ):
            node = ET.SubElement(root, "node")
            node.set("id", node_id)
            node.set("x", f"{x_pos:.2f}")
            node.set("y", f"{y_pos:.2f}")
            node.set("type", node_type)

        ET.ElementTree(root).write(nodes_file, encoding="utf-8", xml_declaration=True)

    def _write_edges_file(self, edges_file: Path) -> None:
        root = ET.Element("edges")
        speed = 13.89  # 50 km/h

        def add_edge(edge_id: str, from_node: str, to_node: str) -> None:
            edge = ET.SubElement(root, "edge")
            edge.set("id", edge_id)
            edge.set("from", from_node)
            edge.set("to", to_node)
            edge.set("priority", "1")
            edge.set("numLanes", "1")
            edge.set("speed", f"{speed:.2f}")

        for edge_id, from_node, to_node in (
            ("north_in", "north", self.tls_id),
            ("south_in", "south", self.tls_id),
            ("east_in", "east", self.tls_id),
            ("west_in", "west", self.tls_id),
            ("north_out", self.tls_id, "south"),
            ("south_out", self.tls_id, "north"),
            ("east_out", self.tls_id, "west"),
            ("west_out", self.tls_id, "east"),
        ):
            add_edge(edge_id, from_node, to_node)

        ET.ElementTree(root).write(edges_file, encoding="utf-8", xml_declaration=True)

    def _write_connections_file(self, connections_file: Path) -> None:
        root = ET.Element("connections")
        for link_index, (from_edge, to_edge) in enumerate(
            zip(self.INCOMING_EDGES, self.OUTGOING_EDGES)
        ):
            conn = ET.SubElement(root, "connection")
            conn.set("from", from_edge)
            conn.set("to", to_edge)
            conn.set("fromLane", "0")
            conn.set("toLane", "0")
            conn.set("dir", "s")  # straight movement
            conn.set("tl", self.tls_id)
            conn.set("linkIndex", str(link_index))

        ET.ElementTree(root).write(connections_file, encoding="utf-8", xml_declaration=True)

    def _extract_lanes_and_update_tl(self, net_file: Path) -> Tuple[List[str], List[str]]:
        tree = ET.parse(net_file)
        root = tree.getroot()

        incoming_lanes: List[str] = []
        outgoing_lanes: List[str] = []

        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if edge_id in self.INCOMING_EDGES or edge_id in self.OUTGOING_EDGES:
                lanes = [lane.get("id", "") for lane in edge.findall("lane")]
                if edge_id in self.INCOMING_EDGES:
                    incoming_lanes.extend(lanes)
                else:
                    outgoing_lanes.extend(lanes)

        logic = None

        # SUMO may place tlLogic either directly under the root or inside <tlLogics>
        for candidate in root.findall("tlLogic"):
            if candidate.get("id") == self.tls_id:
                logic = candidate
                break

        if logic is None:
            tl_logics = root.find("tlLogics")
            if tl_logics is None:
                tl_logics = ET.SubElement(root, "tlLogics")
            for candidate in tl_logics.findall("tlLogic"):
                if candidate.get("id") == self.tls_id:
                    logic = candidate
                    break
            if logic is None:
                logic = ET.SubElement(tl_logics, "tlLogic")
                logic.set("id", self.tls_id)
                logic.set("type", "static")
                logic.set("programID", "0")
                logic.set("offset", "0")

        # Replace existing phases with our two-phase program
        for phase in list(logic.findall("phase")):
            logic.remove(phase)

        for duration, state in (
            (31, "GGrr"),
            (4, "yyrr"),
            (31, "rrGG"),
            (4, "rryy"),
        ):
            phase = ET.SubElement(logic, "phase")
            phase.set("duration", str(duration))
            phase.set("state", state)

        tree.write(net_file, encoding="utf-8", xml_declaration=True)

        return incoming_lanes, outgoing_lanes

    def _resolve_netconvert_binary(self) -> str:
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / "netconvert"
            if candidate.exists():
                return os.fspath(candidate)
        return "netconvert"

    # ------------------------------------------------------------------
    # Route/config writers
    # ------------------------------------------------------------------
    def _write_route_file(self, route_file: Path) -> None:
        root = ET.Element("routes")

        vtype = ET.SubElement(root, "vType")
        vtype.set("id", "car")
        vtype.set("length", "5.0")
        vtype.set("maxSpeed", "13.89")
        vtype.set("accel", "2.5")
        vtype.set("decel", "4.5")
        vtype.set("sigma", "0.5")

        for route_id, edges in (
            ("north_south", "north_in south_out"),
            ("south_north", "south_in north_out"),
            ("east_west", "east_in west_out"),
            ("west_east", "west_in east_out"),
        ):
            route = ET.SubElement(root, "route")
            route.set("id", route_id)
            route.set("edges", edges)

        for flow_id, route_id in (
            ("flow_ns", "north_south"),
            ("flow_sn", "south_north"),
            ("flow_ew", "east_west"),
            ("flow_we", "west_east"),
        ):
            flow = ET.SubElement(root, "flow")
            flow.set("id", flow_id)
            flow.set("type", "car")
            flow.set("route", route_id)
            flow.set("begin", "0")
            flow.set("end", "3600")
            flow.set("departLane", "best")
            flow.set("departSpeed", "max")
            flow.set("vehsPerHour", str(self.flow_rate))

        ET.ElementTree(root).write(route_file, encoding="utf-8", xml_declaration=True)

    def _write_config_file(self, config_file: Path, net_file: Path, route_file: Path) -> None:
        root = ET.Element("configuration")

        input_elem = ET.SubElement(root, "input")
        net_elem = ET.SubElement(input_elem, "net-file")
        net_elem.set("value", os.fspath(net_file))
        route_elem = ET.SubElement(input_elem, "route-files")
        route_elem.set("value", os.fspath(route_file))

        time_elem = ET.SubElement(root, "time")
        ET.SubElement(time_elem, "begin").set("value", "0")
        ET.SubElement(time_elem, "end").set("value", "3600")

        processing_elem = ET.SubElement(root, "processing")
        ET.SubElement(processing_elem, "lateral-resolution").set("value", "0.8")

        report_elem = ET.SubElement(root, "report")
        ET.SubElement(report_elem, "verbose").set("value", "false")
        ET.SubElement(report_elem, "duration-log.disable").set("value", "true")

        ET.ElementTree(root).write(config_file, encoding="utf-8", xml_declaration=True)

    def cleanup(self) -> None:
        """Remove any generated artifacts."""

        if self._artifacts is not None:
            self._artifacts.cleanup()
            self._artifacts = None

