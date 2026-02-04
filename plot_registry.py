from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

@dataclass
class PlotDef:
    id: str
    category: str
    display_name: str
    required_columns: List[str]
    optional_columns: List[str] = field(default_factory=list)
    description: str = ""
    plot_function: Optional[Callable] = None
    supports_grouping: bool = False
    supports_faceting: bool = False

class PlotRegistry:
    _registry: Dict[str, PlotDef] = {}

    @classmethod
    def register(cls, id: str, category: str, display_name: str, required_columns: List[str], 
                 optional_columns: List[str] = [], description: str = "", 
                 supports_grouping: bool = False, supports_faceting: bool = False):
        def decorator(func):
            plot_def = PlotDef(
                id=id,
                category=category,
                display_name=display_name,
                required_columns=required_columns,
                optional_columns=optional_columns,
                description=description,
                plot_function=func,
                supports_grouping=supports_grouping,
                supports_faceting=supports_faceting
            )
            cls._registry[id] = plot_def
            return func
        return decorator

    @classmethod
    def get_all_plots(cls) -> List[PlotDef]:
        return list(cls._registry.values())

    @classmethod
    def get_plot(cls, id: str) -> Optional[PlotDef]:
        return cls._registry.get(id)
        
    @classmethod
    def get_categories(cls) -> List[str]:
        return sorted(list(set(p.category for p in cls._registry.values())))
        
    @classmethod
    def get_plots_by_category(cls, category: str) -> List[PlotDef]:
        return [p for p in cls._registry.values() if p.category == category]
