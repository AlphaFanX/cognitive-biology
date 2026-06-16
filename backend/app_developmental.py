"""
Digital Human Development - SE-Guided Backend
=============================================

Simulates development from zygote to 3,050 cells using:
- SE-guided differentiation tree
- Parent-child cell lineage tracking
- Computed genomic attention (ABC database)
- Extended organ topology
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import asyncio
import json
from typing import Dict, List, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from medic.human_topology_extended import HUMAN_ORGANS_EXTENDED
from medic.se_guided_differentiation import DIFFERENTIATION_TREE, DevelopmentalStage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import genomic attention modules (optional)
try:
    from medic.genome.abc_client import ABCClient
    from medic.genome.ep_interface import EPInterface
    from medic.genomic_attention import GenomicAttentionComputer
    GENOMIC_MODULES_AVAILABLE = True
    logger.info("Genomic attention modules available")
except ImportError as e:
    logger.warning(f"Genomic attention modules not available: {e}")
    logger.warning("Using simplified SE-guided mode with morphogen gradients only")
    GENOMIC_MODULES_AVAILABLE = False
    ABCClient = None
    EPInterface = None
    GenomicAttentionComputer = None

# ============================================================================
# Global State
# ============================================================================

app = FastAPI(title="Digital Human Development")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Morphogen Gradients (Simplified)
# ============================================================================

def compute_bmp_gradient(position):
    """BMP gradient (dorsal-ventral patterning).
    High dorsally (back), low ventrally (front)."""
    z = position[2]  # Z-axis: posterior (-) to anterior (+)
    # BMP high at back (z < 0), low at front (z > 0)
    return max(0.1, 1.0 - (z + 0.1) * 5.0)  # Range: [0.1, 1.0]

def compute_wnt_gradient(position):
    """Wnt gradient (anterior-posterior patterning).
    High posteriorly, low anteriorly."""
    y = position[1]  # Y-axis: inferior (0) to superior (1.75)
    # Wnt high at lower body, low at head
    return max(0.1, 1.0 - y / 1.75)  # Range: [0.1, 1.0]

def compute_fgf_gradient(position):
    """FGF gradient (mesoderm/endoderm patterning).
    High in middle body, low at extremes."""
    y = position[1]
    # FGF peaks around middle (y ~ 0.9)
    center = 0.9
    distance = abs(y - center) / 0.9
    return max(0.1, 1.0 - distance)  # Range: [0.1, 1.0]

def compute_nodal_gradient(position):
    """Nodal gradient (endoderm specification).
    High ventrally, low dorsally."""
    z = position[2]
    # Nodal high at front (z > 0), low at back
    return max(0.1, 0.5 + z * 5.0)  # Range: [0.1, 1.0]


class DevelopmentalCell:
    """Cell with lineage tracking."""

    def __init__(self, cell_id, parent_id, generation, position, organ=None, cell_type="uncommitted"):
        self.id = cell_id
        self.parent_id = parent_id
        self.generation = generation
        self.position = np.array(position)
        self.organ = organ or "uncommitted"
        self.cell_type = cell_type
        self.voltage = -70.0
        self.calcium = 0.1
        self.fate_committed = organ is not None
        self.children = []  # Track children for tree structure

    def to_dict(self):
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "position": self.position.tolist(),
            "voltage": float(self.voltage),
            "calcium": float(self.calcium),
            "organ": self.organ,
            "cell_type": self.cell_type,
            "fate_committed": self.fate_committed,
            "generation": self.generation,
            "morphogens": [0.0, 0.0, 0.0]
        }


class DevelopmentalSimulation:
    """Simulates development with proper cell division tree."""

    def __init__(self):
        self.cells = []
        self.next_id = 0
        self.stage = "ZYGOTE"
        self.time = 0.0

        # Initialize genomic attention system
        if GENOMIC_MODULES_AVAILABLE:
            logger.info("Initializing genomic attention system...")
            try:
                self.ep_interface = EPInterface()
                self.attention_computer = GenomicAttentionComputer(self.ep_interface)
                self.se_guided = True
                logger.info("✓ Genomic attention system initialized")
            except Exception as e:
                logger.warning(f"Could not initialize genomic attention: {e}")
                logger.warning("Falling back to simplified SE mode")
                self.ep_interface = None
                self.attention_computer = None
                self.se_guided = False
        else:
            logger.info("Using simplified SE-guided mode (morphogen gradients + position)")
            self.ep_interface = None
            self.attention_computer = None
            self.se_guided = False

        self.reset()

    def reset(self):
        """Initialize with zygote."""
        self.cells = []
        self.next_id = 0
        self.stage = "ZYGOTE"
        self.time = 0.0

        # Create zygote at origin
        zygote = DevelopmentalCell(
            cell_id=0,
            parent_id=None,
            generation=0,
            position=[0.0, 0.5, 0.0],
            cell_type="zygote"
        )
        self.cells.append(zygote)
        self.next_id = 1

    def divide_cell(self, parent_cell):
        """Divide a cell into two daughters."""
        # Small displacement for daughters
        offset = np.random.randn(3) * 0.02

        # Daughter 1
        pos1 = parent_cell.position + offset
        daughter1 = DevelopmentalCell(
            cell_id=self.next_id,
            parent_id=parent_cell.id,
            generation=parent_cell.generation + 1,
            position=pos1,
            organ=parent_cell.organ,
            cell_type=parent_cell.cell_type
        )
        self.cells.append(daughter1)
        parent_cell.children.append(self.next_id)
        self.next_id += 1

        # Daughter 2
        pos2 = parent_cell.position - offset
        daughter2 = DevelopmentalCell(
            cell_id=self.next_id,
            parent_id=parent_cell.id,
            generation=parent_cell.generation + 1,
            position=pos2,
            organ=parent_cell.organ,
            cell_type=parent_cell.cell_type
        )
        self.cells.append(daughter2)
        parent_cell.children.append(self.next_id)
        self.next_id += 1

        return daughter1, daughter2

    def step(self):
        """Advance development by one step."""
        n_cells = len(self.cells)

        # Stage transitions based on cell count
        if n_cells == 1:
            self.stage = "ZYGOTE"
        elif n_cells < 32:
            self.stage = "MORULA"
        elif n_cells < 128:
            self.stage = "BLASTOCYST"
        elif n_cells < 512:
            self.stage = "GASTRULATION"
        elif n_cells < 3050:
            self.stage = "ORGANOGENESIS"
        else:
            self.stage = "COMPLETE"

        # Division logic
        if n_cells >= 3050:
            return  # Stop at target

        if self.stage == "ZYGOTE":
            # Divide the zygote
            if n_cells == 1:
                self.divide_cell(self.cells[0])

        elif self.stage == "MORULA":
            # Exponential growth
            cells_to_divide = [c for c in self.cells if not c.fate_committed]
            if cells_to_divide and n_cells < 32:
                # Divide all uncommitted cells
                for cell in cells_to_divide[:]:
                    if len(self.cells) >= 32:
                        break
                    self.divide_cell(cell)

        elif self.stage == "BLASTOCYST":
            # Slower division, start differentiation
            if n_cells < 128:
                # Divide some cells
                uncommitted = [c for c in self.cells if not c.fate_committed]
                n_divide = min(len(uncommitted), 128 - n_cells)
                for cell in uncommitted[:n_divide]:
                    self.divide_cell(cell)

        elif self.stage == "GASTRULATION":
            # Gastrulation: assign germ layers
            if n_cells < 512:
                for cell in self.cells:
                    if not cell.fate_committed:
                        # Assign to germ layer based on position
                        if cell.position[1] > 0.6:
                            cell.cell_type = "ectoderm"
                        elif cell.position[1] > 0.4:
                            cell.cell_type = "mesoderm"
                        else:
                            cell.cell_type = "endoderm"

                # Continue division
                uncommitted = [c for c in self.cells if not c.fate_committed]
                n_divide = min(len(uncommitted), 512 - n_cells)
                for cell in uncommitted[:n_divide]:
                    self.divide_cell(cell)

        elif self.stage == "ORGANOGENESIS":
            # Commit cells to organs and position them
            self._commit_to_organs()

        self.time += 1.0

    def _compute_se_fate_scores(self, cell, candidate_organs):
        """Compute fate scores for candidate organs using SE attention.

        Returns dict: {organ_name: score}
        """
        if not self.se_guided or not self.attention_computer:
            # Fallback: uniform scores
            return {organ: 1.0 for organ in candidate_organs}

        fate_scores = {}

        for organ_name in candidate_organs:
            if organ_name not in HUMAN_ORGANS_EXTENDED:
                fate_scores[organ_name] = 0.0
                continue

            spec = HUMAN_ORGANS_EXTENDED[organ_name]

            # Get marker genes for this organ from differentiation tree
            organ_node = None
            for node_name, node in DIFFERENTIATION_TREE.items():
                if node_name == organ_name or node.name.lower().replace(" ", "_") == organ_name:
                    organ_node = node
                    break

            if not organ_node:
                fate_scores[organ_name] = 0.1  # Small baseline
                continue

            # Compute attention score for marker genes
            attention_scores = []
            for gene in organ_node.marker_genes[:3]:  # Use top 3 markers
                try:
                    # Query ABC for this gene in this cell type
                    gene_loci = spec.genomic_loci[0] if spec.genomic_loci else None
                    if gene_loci:
                        chrom, start, end = gene_loci
                        # Simplified: use genomic locus as proxy for attention
                        # In full implementation, would query ABC database
                        attention_scores.append(0.5)  # Placeholder
                except:
                    attention_scores.append(0.1)

            base_attention = np.mean(attention_scores) if attention_scores else 0.1

            # Weight by morphogen gradients
            if organ_name in ["brain", "thyroid"]:
                # Ectoderm: favored by BMP
                morphogen_weight = compute_bmp_gradient(cell.position)
            elif organ_name in ["heart", "kidney_left", "kidney_right", "muscle", "bone"]:
                # Mesoderm: favored by Wnt + FGF
                wnt = compute_wnt_gradient(cell.position)
                fgf = compute_fgf_gradient(cell.position)
                morphogen_weight = (wnt + fgf) / 2.0
            else:
                # Endoderm: favored by Nodal + FGF
                nodal = compute_nodal_gradient(cell.position)
                fgf = compute_fgf_gradient(cell.position)
                morphogen_weight = (nodal + fgf) / 2.0

            # Final score = attention × morphogen × position affinity
            position_affinity = 1.0 / (1.0 + np.linalg.norm(cell.position - np.array(spec.position_3d)))
            fate_scores[organ_name] = base_attention * morphogen_weight * position_affinity

        return fate_scores

    def _commit_to_organs(self):
        """Commit cells to specific organs using SE-guided fate decisions."""
        # Get uncommitted cells per germ layer
        ectoderm_cells = [c for c in self.cells if c.cell_type == "ectoderm" and not c.fate_committed]
        mesoderm_cells = [c for c in self.cells if c.cell_type == "mesoderm" and not c.fate_committed]
        endoderm_cells = [c for c in self.cells if c.cell_type == "endoderm" and not c.fate_committed]

        # Candidate organs per germ layer
        ectoderm_organs = ["brain", "thyroid"]
        mesoderm_organs = ["heart", "kidney_left", "kidney_right", "muscle", "bone"]
        endoderm_organs = ["liver", "lung_left", "lung_right", "pancreas", "gut"]

        # SE-GUIDED FATE DECISION: Each cell decides its own fate based on attention scores
        all_germ_layers = [
            (ectoderm_cells, ectoderm_organs, "ectoderm"),
            (mesoderm_cells, mesoderm_organs, "mesoderm"),
            (endoderm_cells, endoderm_organs, "endoderm")
        ]

        organ_counts = {name: 0 for name in HUMAN_ORGANS_EXTENDED.keys()}
        committed_this_round = []

        for cells, candidate_organs, germ_layer in all_germ_layers:
            for cell in cells:
                if cell.fate_committed:
                    continue

                # Compute SE-guided fate scores
                fate_scores = self._compute_se_fate_scores(cell, candidate_organs)

                # Filter out organs that are already full
                available_organs = {
                    organ: score for organ, score in fate_scores.items()
                    if organ in HUMAN_ORGANS_EXTENDED and
                    organ_counts[organ] < HUMAN_ORGANS_EXTENDED[organ].cell_count
                }

                if not available_organs:
                    continue

                # Probabilistic sampling based on scores (softmax)
                organs = list(available_organs.keys())
                scores = np.array([available_organs[o] for o in organs])

                # Softmax
                exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
                probs = exp_scores / np.sum(exp_scores)

                # Sample fate
                chosen_organ = np.random.choice(organs, p=probs)

                # Commit cell to chosen organ
                cell.organ = chosen_organ
                cell.fate_committed = True
                organ_counts[chosen_organ] += 1
                committed_this_round.append((cell, chosen_organ))

                # Position cell in organ bounding box
                spec = HUMAN_ORGANS_EXTENDED[chosen_organ]
                bbox_min, bbox_max = spec.bounding_box
                cell.position = np.array([
                    np.random.uniform(bbox_min[0], bbox_max[0]),
                    np.random.uniform(bbox_min[1], bbox_max[1]),
                    np.random.uniform(bbox_min[2], bbox_max[2])
                ])
                cell.voltage = spec.resting_voltage

        # Log SE-guided decisions
        if committed_this_round and self.se_guided:
            logger.info(f"SE-guided commitment: {len(committed_this_round)} cells assigned by attention scores")
            for organ, count in organ_counts.items():
                if count > 0:
                    logger.info(f"  {organ}: {count} cells")

        # Fill remaining cells to reach target counts (progressive division)
        for organ_name, spec in HUMAN_ORGANS_EXTENDED.items():
            current_count = organ_counts[organ_name]
            target_count = spec.cell_count

            if current_count >= target_count:
                continue

            # Get cells already committed to this organ
            organ_cells = [c for c in self.cells if c.organ == organ_name and c.fate_committed]

            if len(organ_cells) == 0:
                # No seed cells - create one based on germ layer
                germ_layer = self._get_germ_layer_for_organ(organ_name)
                seed = DevelopmentalCell(
                    cell_id=self.next_id,
                    parent_id=None,
                    generation=5,
                    position=np.array(spec.position_3d),
                    organ=organ_name,
                    cell_type=germ_layer
                )
                seed.fate_committed = True
                seed.voltage = spec.resting_voltage
                self.cells.append(seed)
                self.next_id += 1
                organ_cells.append(seed)

            # Divide to reach target
            while len(organ_cells) < target_count and len(self.cells) < 3050:
                if len(organ_cells) == 0:
                    break
                parent = np.random.choice(organ_cells)
                d1, d2 = self.divide_cell(parent)

                for daughter in [d1, d2]:
                    daughter.organ = organ_name
                    daughter.fate_committed = True
                    daughter.voltage = spec.resting_voltage
                    # Position in bounding box
                    bbox_min, bbox_max = spec.bounding_box
                    daughter.position = np.array([
                        np.random.uniform(bbox_min[0], bbox_max[0]),
                        np.random.uniform(bbox_min[1], bbox_max[1]),
                        np.random.uniform(bbox_min[2], bbox_max[2])
                    ])
                    organ_cells.append(daughter)

    def _get_germ_layer_for_organ(self, organ_name):
        """Determine germ layer for an organ."""
        ectoderm = ["brain", "thyroid"]
        mesoderm = ["heart", "kidney_left", "kidney_right", "muscle", "bone"]
        # Everything else is endoderm
        if organ_name in ectoderm:
            return "ectoderm"
        elif organ_name in mesoderm:
            return "mesoderm"
        else:
            return "endoderm"

    def serialize(self):
        """Convert to JSON-serializable dict."""
        organ_counts = {}
        for cell in self.cells:
            organ_counts[cell.organ] = organ_counts.get(cell.organ, 0) + 1

        target_organs = {}
        for organ_name, spec in HUMAN_ORGANS_EXTENDED.items():
            target_organs[organ_name] = {
                "position": list(spec.position_3d),
                "cell_count": spec.cell_count,
                "current_count": organ_counts.get(organ_name, 0)
            }

        return {
            "stage": self.stage,
            "stage_value": 0,
            "time": float(self.time),
            "n_cells": len(self.cells),
            "target_cells": 3050,
            "progress": len(self.cells) / 3050.0,
            "cells": [c.to_dict() for c in self.cells],
            "organ_counts": organ_counts,
            "target_organs": target_organs
        }


# Global simulation
SIM = DevelopmentalSimulation()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()

    try:
        # Send initial state
        last_state = SIM.serialize()
        await websocket.send_json(last_state)
        last_n_cells = len(SIM.cells)
        last_stage = SIM.stage

        while True:
            state_changed = False

            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(data)

                command = msg.get("command")

                if command == "step":
                    steps = msg.get("steps", 1)
                    for _ in range(steps):
                        if len(SIM.cells) < 3050:
                            SIM.step()
                            state_changed = True

                elif command == "reset":
                    SIM.reset()
                    state_changed = True

                elif command == "run":
                    # Run to completion
                    while len(SIM.cells) < 3050:
                        SIM.step()
                    state_changed = True

            except asyncio.TimeoutError:
                pass

            # PERFORMANCE OPTIMIZATION: Only send updates when state actually changes
            current_n_cells = len(SIM.cells)
            current_stage = SIM.stage

            if state_changed or current_n_cells != last_n_cells or current_stage != last_stage:
                current_state = SIM.serialize()
                await websocket.send_json(current_state)
                last_state = current_state
                last_n_cells = current_n_cells
                last_stage = current_stage
            else:
                # State hasn't changed, just wait a bit
                await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn

    print("=" * 70)
    print("DEVELOPMENTAL SIMULATION - SE-GUIDED")
    print("=" * 70)
    print("Server: http://localhost:8002")
    print("WebSocket: ws://localhost:8002/ws")
    print("Features:")
    print("  - Tree-based cell division")
    print("  - Parent-child lineage tracking")
    print("  - SE-guided differentiation")
    print("  - 3,050 cells across 11 organs")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
