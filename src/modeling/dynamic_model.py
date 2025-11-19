import torch.nn as nn
import torch
from src.modeling.shared_components import SHARED_COMPONENTS
from datetime import datetime, timezone
from pathlib import Path
from src.config import Config

def assemble_model(parts_from):
    """Assemble a model from the shared components.

    Args:
        SHARED_COMPONENTS (dict): A dictionary containing the shared components.
        REQUIRED_COMPONENT_NAMES (list): A list containing the names of the required components.

    Returns:
        DynamicModel: A model assembled from the shared components.
    """
    # if internal tuple shape is of len 1, then it is a list
    if isinstance(parts_from[0], str):
        part_paths = dict(zip(parts_from, tuple([None] * len(parts_from))))
    elif isinstance(parts_from[0], tuple) and len(parts_from[0])==4:
        checpoint_dir = Path(Config["DIRS"]["OUTPUT"]["CHECKPOINTS"])
        part_paths = {}
        for part, entity, project, run_id in parts_from:
            if type(part)==str and type(entity)==str and type(project)==str and type(run_id)==str:
                if run_id == "latest":
                    latest_run_dir = get_latest_run_dir(checpoint_dir / entity / project)
                    part_paths[part] = latest_run_dir / part
                else:
                    part_paths[part] = checpoint_dir / entity / project / run_id / part
            else:
                part_paths[part] = None
    else:
        raise ValueError(f"Invalid parts_from shape: {len(parts_from[0])}")
    components = []
    names = []
    for name, part_dir in part_paths.items():
        component = SHARED_COMPONENTS.get(name)
        if isinstance(part_dir, Path):
            filepath = get_latest_part(part_dir)
            state_dict = torch.load(filepath)
            component.load_state_dict(state_dict)
        elif part_dir is not None:
            raise ValueError(f"Invalid run_dir type. Must be of type Path but is of type {type(part_dir)}\n\nrun_dir: {part_dir}")
        if component is None:
            raise KeyError(f"Component {name} not found in shared components.")
        components.append(component)
        names.append(name)
    return DynamicModel(components, names)

def get_latest_run_dir(project_dir):
    """Get the latest run directory for a project.

    Args:
        project_dir (str): The directory where the runs are stored.

    Returns:
        str: The name of the latest run directory.
    """
    run_dirs = list(project_dir.glob("*"))
    if len(run_dirs) == 0:
        raise ValueError(f"No runs found in {project_dir}")
    run_dirs = [run_dir for run_dir in run_dirs if run_dir.is_dir()]
    if len(run_dirs) == 0:
        raise ValueError(f"No runs found in {project_dir}")
    # sort by last created (CREATED NOT MODIFIED)
    run_dirs.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    return run_dirs[0]

def get_latest_part(part_dir):
    """Get the latest part for a component.

    Args:
        name (str): The name of the component.
        run_dir (str): The directory where the component parts are stored.

    Returns:
        str: The filename of the latest part.
    """
    part_paths = list(part_dir.glob(f"*.pt"))
    if len(part_paths) == 0:
        raise ValueError(f"No parts found in {part_dir}")
    part_paths.sort(key=lambda x: x.stat().st_ctime, reverse=True)
    return part_paths[0]


class DynamicModel(nn.Module):
    def __init__(self, components, component_names):
        super(DynamicModel, self).__init__()
        self.components = nn.ModuleList(components)
        self.component_names = component_names
        # make a new attribute by the name of each component and assign the component to it
        for component, name in zip(components, component_names):
            setattr(self, name, component)
        # get current torch device (cpu or gpu) that was previously set
        self.device = next(self.parameters()).device
        # check that all the components can compose together
        for i in range(len(components) - 1):
            component1 = components[i]
            component2 = components[i + 1]
            if component1.output_shape != component2.input_shape:
                raise ValueError(f"Component {i} output shape {component1.output_shape} does not match component {i + 1} input shape {component2.input_shape}.")

    def forward(self, data: dict):
        result = None
        for i, component in enumerate(self.components):
            if i == 0:
                result = component(data["input"].to(self.device))
            else:
                result = component(result)
                if component.is_output_layer:
                    data[component.name] = result
        return data

    def save(self, path, epoch_n, batch_n):
        now = datetime.now(timezone.utc)
        filename = now.strftime("%d-%m-%Y_%H-%M-%S") + f'_UTC-epoch_{epoch_n}-batch_{batch_n}.pt'
        for component, name in zip(self.components, self.component_names):
            component_path = path / name / filename
            component_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(component.state_dict(), component_path)

    def load(self, directory, device='cpu'):
        """Loads each component of the model from a specified directory."""
        for component, name in zip(self.components, self.component_names):
            component_dir = Path(directory) / name
            # Find the latest saved file
            latest_file = max(component_dir.glob('*.pt'), key=lambda x: x.stat().st_mtime)
            component.load_state_dict(torch.load(latest_file, map_location=device))
            print(f'Loaded {name} from {latest_file}')
