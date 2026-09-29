"""Competition registry — maps app menu categories to LeagueProfile objects."""
from src.config import LEAGUE_REGISTRY, LeagueProfile


def get_league(name: str) -> LeagueProfile:
    """Fetch a league profile by name, case-insensitive."""
    for key, profile in LEAGUE_REGISTRY.items():
        if key.lower() == name.lower():
            return profile
    raise KeyError(f"League '{name}' not found in registry. "
                   f"Available: {list(LEAGUE_REGISTRY.keys())}")


def list_leagues_by_country(country: str) -> list[tuple[str, LeagueProfile]]:
    """Return all leagues for a given country."""
    return [(k, v) for k, v in LEAGUE_REGISTRY.items()
            if v.country.lower() == country.lower()]


def list_all_countries() -> list[str]:
    """Return sorted unique countries."""
    return sorted(set(p.country for p in LEAGUE_REGISTRY.values()))


def list_all_leagues() -> list[str]:
    """Return sorted league names."""
    return sorted(LEAGUE_REGISTRY.keys())
