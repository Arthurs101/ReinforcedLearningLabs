"""Scenario builders for SUMO traffic simulations."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict


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
        flow_rate: int = 1000,  # Default: 1000 vehicles/hour (moderate traffic)
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


class OSMScenario:
    """Convert an OSM map file to SUMO network format and generate routes."""

    def __init__(
        self,
        osm_file: Path,
        flow_rate: int = 1000,  # Default: 1000 vehicles/hour (moderate traffic)
        output_dir: Optional[Path] = None,
        tls_id: Optional[str] = None,
        begin_time: int = 0,
        end_time: int = 3600,
        netconvert_options: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize OSM scenario builder.

        Args:
            osm_file: Path to the OSM file
            flow_rate: Vehicle flow rate per hour
            output_dir: Directory to store generated files (default: temp directory)
            tls_id: Traffic light ID to control (default: first TLS found in network)
            begin_time: Simulation start time
            end_time: Simulation end time
            netconvert_options: Additional options for netconvert
        """
        self.osm_file = Path(osm_file)
        if not self.osm_file.exists():
            raise FileNotFoundError(f"OSM file not found: {self.osm_file}")
        
        self.flow_rate = flow_rate
        self.begin_time = begin_time
        self.end_time = end_time
        self.tls_id = tls_id
        self._provided_dir = output_dir
        self._artifacts: Optional[ScenarioArtifacts] = None
        self._netconvert_options = netconvert_options or []

    def build(self, regenerate: bool = False) -> ScenarioArtifacts:
        """Create the SUMO network, routes, and configuration files from OSM."""

        if self._artifacts is not None and not regenerate:
            return self._artifacts

        temp_dir = Path(self._provided_dir) if self._provided_dir else Path(
            tempfile.mkdtemp(prefix="sumo-osm-")
        )
        temp_dir.mkdir(parents=True, exist_ok=True)

        net_file = temp_dir / "network.net.xml"
        route_file = temp_dir / "routes.rou.xml"
        config_file = temp_dir / "config.sumocfg"
        trips_file = temp_dir / "trips.trips.xml"

        # Convert OSM to SUMO network
        tls_id, plain_files = self._generate_network(temp_dir, net_file)

        # Get all available TLS IDs for validation
        all_tls_ids = self._get_all_tls_ids(net_file)
        
        # Use provided TLS ID or the first one found
        if self.tls_id:
            if self.tls_id not in all_tls_ids:
                print(f"\nWarning: Specified traffic light ID '{self.tls_id}' not found in network!", file=sys.stderr)
                if all_tls_ids:
                    print(f"Available traffic light IDs (showing first 10): {', '.join(all_tls_ids[:10])}", file=sys.stderr)
                    if len(all_tls_ids) > 10:
                        print(f"... and {len(all_tls_ids) - 10} more", file=sys.stderr)
                    print(f"Using first available TLS ID: {all_tls_ids[0]}", file=sys.stderr)
                    final_tls_id = all_tls_ids[0]
                else:
                    raise ValueError("No traffic lights found in the network. Please ensure the OSM file contains traffic signals.")
            else:
                final_tls_id = self.tls_id
        else:
            final_tls_id = tls_id
            if not final_tls_id:
                raise ValueError("No traffic lights found in the network. Please ensure the OSM file contains traffic signals.")

        # Extract lanes for the selected traffic light
        incoming_lanes, outgoing_lanes = self._extract_lanes_for_tls(net_file, final_tls_id)
        
        if not incoming_lanes:
            print(f"Warning: No incoming lanes found for traffic light '{final_tls_id}'.", file=sys.stderr)
            print("This may indicate the traffic light is not properly connected in the network.", file=sys.stderr)

        # Generate routes
        self._generate_routes(net_file, trips_file, route_file)

        # Write config file
        self._write_config_file(config_file, net_file, route_file)

        self._artifacts = ScenarioArtifacts(
            directory=temp_dir,
            net_file=net_file,
            route_file=route_file,
            config_file=config_file,
            tls_id=final_tls_id,
            incoming_lanes=incoming_lanes,
            outgoing_lanes=outgoing_lanes,
            plain_files=plain_files + [trips_file],
        )
        return self._artifacts

    def _generate_network(
        self, temp_dir: Path, net_file: Path
    ) -> Tuple[Optional[str], List[Path]]:
        """Convert OSM file to SUMO network using netconvert with cleaning options."""
        
        # Try to pre-filter OSM file to only include roads and traffic signals
        filtered_osm = self._filter_osm_file(temp_dir)
        osm_input = filtered_osm if filtered_osm else self.osm_file
        
        cmd = [
            self._resolve_netconvert_binary(),
            "--osm-files",
            os.fspath(osm_input),
            "--output-file",
            os.fspath(net_file),
            "--no-internal-links",
            "--geometry.remove",
            "--roundabouts.guess",
            "true",
            "--ramps.guess",
            "true",
            "--junctions.join",
            "true",
            "--tls.guess-signals",
            "true",
            "--tls.discard-simple",
            "true",
            "--tls.join",
            "true",
            # Simplify geometry to avoid angle calculation problems
            "--geometry.max-angle",
            "99.0",
            "--junctions.corner-detail",
            "0",
            # Additional robustness options
            "--junctions.limit-turn-speed",
            "5.5",
            "--edges.join",
            "true",
            # Filter out non-road vehicle classes after conversion
            "--remove-edges.by-vclass",
            "rail_slow,rail_fast,bicycle,pedestrian,ship",
        ]
        cmd.extend(self._netconvert_options)

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # If it fails, try with even more relaxed options (less aggressive cleaning)
            error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else "Unknown error")
            print("Warning: Initial netconvert failed, retrying with relaxed options...", file=sys.stderr)
            if error_msg:
                # Print relevant error info (skip verbose warnings)
                error_lines = error_msg.split('\n')
                critical_errors = [line for line in error_lines if 'Error:' in line or 'Assertion' in line or 'failed' in line.lower()]
                if critical_errors:
                    print("Critical errors:", file=sys.stderr)
                    for err in critical_errors[:3]:  # Show first 3 critical errors
                        print(f"  {err}", file=sys.stderr)
            
            # Fallback: minimal options to avoid angle calculation issues
            cmd_fallback = [
                self._resolve_netconvert_binary(),
                "--osm-files",
                os.fspath(osm_input),
                "--output-file",
                os.fspath(net_file),
                "--no-internal-links",
                "--geometry.remove",
                "--roundabouts.guess",
                "true",
                "--tls.guess-signals",
                "true",
                "--geometry.max-angle",
                "99.0",
                "--junctions.corner-detail",
                "0",
                "--remove-edges.by-vclass",
                "rail_slow,rail_fast,bicycle,pedestrian,ship",
            ]
            cmd_fallback.extend(self._netconvert_options)
            
            try:
                subprocess.run(cmd_fallback, check=True)
            except subprocess.CalledProcessError as e2:
                # Last resort: absolute minimum options, no roundabout processing
                print("Warning: Fallback also failed, trying minimal options without roundabout processing...", file=sys.stderr)
                cmd_minimal = [
                    self._resolve_netconvert_binary(),
                    "--osm-files",
                    os.fspath(osm_input),
                    "--output-file",
                    os.fspath(net_file),
                    "--no-internal-links",
                    "--geometry.remove",
                    # Skip roundabout processing which triggers angle calculation
                    "--roundabouts.guess",
                    "false",
                    "--tls.guess-signals",
                    "true",
                    "--remove-edges.by-vclass",
                    "rail_slow,rail_fast,bicycle,pedestrian,ship",
                ]
                cmd_minimal.extend(self._netconvert_options)
                
                try:
                    subprocess.run(cmd_minimal, check=True)
                except subprocess.CalledProcessError as e3:
                    # Final attempt: absolute bare minimum
                    print("\n" + "="*80, file=sys.stderr)
                    print("ERROR: All netconvert conversion attempts failed!", file=sys.stderr)
                    print("="*80, file=sys.stderr)
                    print("The OSM file contains problematic junction geometry that causes netconvert", file=sys.stderr)
                    print("to crash with an angle calculation assertion failure.", file=sys.stderr)
                    print("\nThis is NOT about roundabouts - it's a bug in netconvert when processing", file=sys.stderr)
                    print("certain complex or malformed junctions in the OSM data.", file=sys.stderr)
                    print("\nPossible solutions:", file=sys.stderr)
                    print("1. Pre-process the OSM file to fix problematic junctions", file=sys.stderr)
                    print("2. Use a different map region/extract with cleaner geometry", file=sys.stderr)
                    print("3. Use SUMO's polyconvert or osmFilter to clean the OSM data first", file=sys.stderr)
                    print("4. Report this as a bug to the SUMO project with the OSM file", file=sys.stderr)
                    print("="*80 + "\n", file=sys.stderr)
                    raise RuntimeError(
                        f"Failed to convert OSM file '{self.osm_file}' to SUMO network. "
                        "The file contains problematic geometry that causes netconvert to crash. "
                        "See error messages above for details."
                    ) from e3

        # Extract traffic light IDs
        tls_id = self._extract_tls_ids(net_file)

        return tls_id, []

    def _filter_osm_file(self, temp_dir: Path) -> Optional[Path]:
        """Pre-filter OSM file to only include roads and traffic signals using osmfilter if available."""
        try:
            # Check if osmfilter is available
            result = subprocess.run(
                ["osmfilter", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return None  # osmfilter not available
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None  # osmfilter not found
        
        # Filter OSM file to only keep roads and traffic signals
        filtered_osm = temp_dir / "filtered.osm"
        cmd = [
            "osmfilter",
            os.fspath(self.osm_file),
            "--keep=",
            "highway=motorway =motorway_link =trunk =trunk_link =primary =primary_link "
            "=secondary =secondary_link =tertiary =tertiary_link =residential "
            "=unclassified =service =living_street =pedestrian =track",
            "--keep-nodes=",
            "highway=traffic_signals =stop =give_way",
            "--keep-ways=",
            "highway=",
            "-o=",
            os.fspath(filtered_osm),
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            if filtered_osm.exists() and filtered_osm.stat().st_size > 0:
                print("Pre-filtered OSM file to only include roads and traffic signals", file=sys.stderr)
                return filtered_osm
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass  # If filtering fails, use original file
        
        return None

    def _get_all_tls_ids(self, net_file: Path) -> List[str]:
        """Get all traffic light IDs from the network file."""
        tree = ET.parse(net_file)
        root = tree.getroot()

        tls_ids: List[str] = []

        # Find all traffic lights
        for tl_logic in root.findall("tlLogic"):
            tls_id = tl_logic.get("id")
            if tls_id:
                tls_ids.append(tls_id)

        # If no tlLogic elements found, check inside tlLogics container
        if not tls_ids:
            tl_logics = root.find("tlLogics")
            if tl_logics is not None:
                for tl_logic in tl_logics.findall("tlLogic"):
                    tls_id = tl_logic.get("id")
                    if tls_id:
                        tls_ids.append(tls_id)

        # If still no TLS found, check junctions
        if not tls_ids:
            for junction in root.findall("junction"):
                junction_id = junction.get("id", "")
                junction_type = junction.get("type", "")
                if junction_type == "traffic_light":
                    tls_ids.append(junction_id)

        return tls_ids

    def _extract_tls_ids(self, net_file: Path) -> Optional[str]:
        """Extract the first traffic light ID from the network file."""
        tls_ids = self._get_all_tls_ids(net_file)
        return tls_ids[0] if tls_ids else None

    def _extract_lanes_for_tls(
        self, net_file: Path, tls_id: str
    ) -> Tuple[List[str], List[str]]:
        """Extract lanes connected to a specific traffic light."""
        tree = ET.parse(net_file)
        root = tree.getroot()

        incoming_lanes: List[str] = []
        outgoing_lanes: List[str] = []

        # Find junction nodes controlled by this TLS
        tls_nodes: set[str] = set()
        
        # Check if TLS ID matches a junction directly
        for junction in root.findall("junction"):
            junction_id = junction.get("id", "")
            junction_type = junction.get("type", "")
            if junction_type == "traffic_light" and (junction_id == tls_id or junction_id.startswith(tls_id)):
                tls_nodes.add(junction_id)

        # Also check connections to find lanes controlled by this TLS
        for connection in root.findall("connection"):
            tl_attr = connection.get("tl", "")
            if tl_attr == tls_id:
                from_edge_id = connection.get("from", "")
                from_lane_idx = int(connection.get("fromLane", "0"))
                # Find the actual lane ID from the edge
                for edge in root.findall("edge"):
                    if edge.get("id") == from_edge_id:
                        lanes = edge.findall("lane")
                        if from_lane_idx < len(lanes):
                            lane_id = lanes[from_lane_idx].get("id", "")
                            if lane_id and lane_id not in incoming_lanes:
                                incoming_lanes.append(lane_id)
                        break

        # Find edges connected to traffic light junctions
        for edge in root.findall("edge"):
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            
            # Check if edge is connected to a traffic light junction
            if to_node in tls_nodes:
                for lane in edge.findall("lane"):
                    lane_id = lane.get("id", "")
                    if lane_id and lane_id not in incoming_lanes:
                        incoming_lanes.append(lane_id)
            
            if from_node in tls_nodes:
                for lane in edge.findall("lane"):
                    lane_id = lane.get("id", "")
                    if lane_id and lane_id not in outgoing_lanes:
                        outgoing_lanes.append(lane_id)

        return incoming_lanes, outgoing_lanes

    def get_tls_positions(self, net_file: Path) -> Dict[str, Tuple[float, float]]:
        """Get positions (x, y) of all traffic lights from the network file.
        
        Returns:
            Dictionary mapping TLS ID to (x, y) position tuple.
        """
        tree = ET.parse(net_file)
        root = tree.getroot()
        
        tls_positions: Dict[str, Tuple[float, float]] = {}
        
        # Get TLS IDs
        tls_ids = self._get_all_tls_ids(net_file)
        
        # Find positions from junctions
        for junction in root.findall("junction"):
            junction_id = junction.get("id", "")
            junction_type = junction.get("type", "")
            
            if junction_type == "traffic_light" and junction_id in tls_ids:
                x = float(junction.get("x", "0"))
                y = float(junction.get("y", "0"))
                tls_positions[junction_id] = (x, y)
        
        # Also check tlLogic elements for positions
        for tl_logic in root.findall("tlLogic"):
            tls_id = tl_logic.get("id")
            if tls_id and tls_id not in tls_positions:
                # Try to find associated junction
                for junction in root.findall("junction"):
                    if junction.get("id") == tls_id:
                        x = float(junction.get("x", "0"))
                        y = float(junction.get("y", "0"))
                        tls_positions[tls_id] = (x, y)
                        break
        
        return tls_positions

    def group_tls_by_proximity(
        self,
        net_file: Path,
        tls_ids: Optional[List[str]] = None,
        max_tls_per_group: int = 4,
        max_distance: Optional[float] = None,
    ) -> List[List[str]]:
        """Group traffic lights by proximity.
        
        Args:
            net_file: Path to network file
            tls_ids: List of TLS IDs to group. If None, uses all TLS.
            max_tls_per_group: Maximum number of TLS per group
            max_distance: Maximum distance (meters) for grouping. If None, uses clustering.
        
        Returns:
            List of TLS ID groups, where each group is a list of TLS IDs.
        """
        if tls_ids is None:
            tls_ids = list(self.get_all_tls_lanes(net_file).keys())
        
        if not tls_ids:
            return []
        
        # Get positions
        positions = self.get_tls_positions(net_file)
        
        # Filter TLS with known positions
        tls_with_positions = [(tls_id, positions.get(tls_id)) for tls_id in tls_ids if tls_id in positions]
        
        if not tls_with_positions:
            # If no positions found, use simple sequential grouping
            groups = []
            for i in range(0, len(tls_ids), max_tls_per_group):
                groups.append(tls_ids[i:i + max_tls_per_group])
            return groups
        
        # Group by proximity using simple distance-based clustering
        groups: List[List[str]] = []
        used_tls = set()
        
        for tls_id, pos in tls_with_positions:
            if tls_id in used_tls:
                continue
            
            if pos is None:
                # TLS without position gets its own group
                groups.append([tls_id])
                used_tls.add(tls_id)
                continue
            
            # Start a new group
            group = [tls_id]
            used_tls.add(tls_id)
            
            # Find nearby TLS
            for other_tls_id, other_pos in tls_with_positions:
                if other_tls_id in used_tls or other_pos is None:
                    continue
                
                if len(group) >= max_tls_per_group:
                    break
                
                # Calculate distance
                distance = ((pos[0] - other_pos[0]) ** 2 + (pos[1] - other_pos[1]) ** 2) ** 0.5
                
                if max_distance is None or distance <= max_distance:
                    group.append(other_tls_id)
                    used_tls.add(other_tls_id)
            
            groups.append(group)
        
        # Add any TLS without positions as individual groups
        for tls_id in tls_ids:
            if tls_id not in used_tls:
                groups.append([tls_id])
        
        return groups

    def get_all_tls_lanes(self, net_file: Path) -> Dict[str, Tuple[List[str], List[str]]]:
        """Get all traffic light IDs and their associated lanes.
        
        Returns:
            Dictionary mapping TLS ID to (incoming_lanes, outgoing_lanes) tuple.
        """
        all_tls_ids = self._get_all_tls_ids(net_file)
        tls_lanes: Dict[str, Tuple[List[str], List[str]]] = {}
        
        for tls_id in all_tls_ids:
            incoming_lanes, outgoing_lanes = self._extract_lanes_for_tls(net_file, tls_id)
            if incoming_lanes:  # Only include TLS with incoming lanes
                tls_lanes[tls_id] = (incoming_lanes, outgoing_lanes)
        
        return tls_lanes

    def _generate_routes(
        self, net_file: Path, trips_file: Path, route_file: Path
    ) -> None:
        """Generate routes using randomTrips.py and duarouter."""
        
        # First, generate trips using randomTrips.py
        random_trips_script = self._resolve_random_trips_binary()
        python_exe = sys.executable  # Use the same Python interpreter
        
        random_trips_cmd = [
            python_exe,
            random_trips_script,
            "-n",
            os.fspath(net_file),
            "-o",
            os.fspath(trips_file),
            "-b",
            str(self.begin_time),
            "-e",
            str(self.end_time),
            "--insertion-rate",
            str(self.flow_rate),  # Vehicles per hour
            "--random",
        ]

        subprocess.run(random_trips_cmd, check=True)

        # Then, convert trips to routes using duarouter
        duarouter_cmd = [
            self._resolve_duarouter_binary(),
            "-n",
            os.fspath(net_file),
            "-t",
            os.fspath(trips_file),
            "-o",
            os.fspath(route_file),
            "--ignore-errors",
            "--remove-loops",
        ]

        subprocess.run(duarouter_cmd, check=True)

    def _write_config_file(
        self, config_file: Path, net_file: Path, route_file: Path
    ) -> None:
        """Write SUMO configuration file."""
        root = ET.Element("configuration")

        input_elem = ET.SubElement(root, "input")
        net_elem = ET.SubElement(input_elem, "net-file")
        net_elem.set("value", os.fspath(net_file))
        route_elem = ET.SubElement(input_elem, "route-files")
        route_elem.set("value", os.fspath(route_file))

        time_elem = ET.SubElement(root, "time")
        ET.SubElement(time_elem, "begin").set("value", str(self.begin_time))
        ET.SubElement(time_elem, "end").set("value", str(self.end_time))

        processing_elem = ET.SubElement(root, "processing")
        ET.SubElement(processing_elem, "lateral-resolution").set("value", "0.8")

        report_elem = ET.SubElement(root, "report")
        ET.SubElement(report_elem, "verbose").set("value", "false")
        ET.SubElement(report_elem, "duration-log.disable").set("value", "true")

        ET.ElementTree(root).write(config_file, encoding="utf-8", xml_declaration=True)

    def _resolve_netconvert_binary(self) -> str:
        """Resolve path to netconvert binary."""
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / "netconvert"
            if candidate.exists():
                return os.fspath(candidate)
        return "netconvert"

    def _resolve_duarouter_binary(self) -> str:
        """Resolve path to duarouter binary."""
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = Path(sumo_home) / "bin" / "duarouter"
            if candidate.exists():
                return os.fspath(candidate)
        return "duarouter"

    def _resolve_random_trips_binary(self) -> str:
        """Resolve path to randomTrips.py script."""
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home:
            candidate = Path(sumo_home) / "tools" / "randomTrips.py"
            if candidate.exists():
                return os.fspath(candidate)
        return "randomTrips.py"

    def cleanup(self) -> None:
        """Remove any generated artifacts."""
        if self._artifacts is not None:
            self._artifacts.cleanup()
            self._artifacts = None

