"""
Excel Parser Module
Parses Risk Analysis Excel files and extracts structured data for graph ingestion.
Implements full scope requirements including:
- Hazard ↔ Cause ↔ Consequence chains
- Control reuse across hazards
- Risk reduction before and after mitigation
- Contextual links to standards, instructions, and lifecycle stages
- Shared controls, conditional mitigations, and residual risk dependencies
"""
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Hazard:
    id: str
    name: str
    h_type: str = ""
    q_source: str = ""
    p_init: Optional[int] = None
    s_init: Optional[int] = None
    r_init: str = ""
    p_final: Optional[int] = None
    s_final: Optional[int] = None
    r_final: str = ""


@dataclass
class Cause:
    id: str
    description: str
    hazard_id: str


@dataclass
class Consequence:
    id: str
    description: str
    hazard_id: str


@dataclass
class Control:
    id: str
    description: str
    implementation_reference: str = ""


@dataclass
class HazardCategory:
    id: str
    name: str


@dataclass
class Actor:
    name: str


@dataclass
class Standard:
    """Standards referenced by controls (e.g., IEC 61010-1, EMC 61326)"""
    id: str
    name: str


@dataclass
class DocumentSection:
    """Document sections where controls are documented (e.g., OI Section, Maintenance)"""
    id: str
    name: str
    document_type: str = ""  # OI, PSS, QMS, etc.


@dataclass
class LifecyclePhase:
    """Lifecycle phases when hazards occur (e.g., Operation, Maintenance, Installation)"""
    name: str


@dataclass
class HazardControlLink:
    hazard_id: str
    control_id: str
    p_reduction: Optional[int] = None  # Probability reduction
    s_reduction: Optional[int] = None  # Severity reduction


@dataclass
class HazardActorLink:
    hazard_id: str
    actor_name: str


@dataclass
class HazardCategoryLink:
    hazard_id: str
    category_id: str


@dataclass
class HazardLifecycleLink:
    hazard_id: str
    lifecycle_phase: str


@dataclass
class ControlStandardLink:
    control_id: str
    standard_id: str


@dataclass
class ControlDocumentLink:
    control_id: str
    document_id: str


@dataclass
class ParsedData:
    hazard_categories: List[HazardCategory] = field(default_factory=list)
    hazards: List[Hazard] = field(default_factory=list)
    causes: List[Cause] = field(default_factory=list)
    consequences: List[Consequence] = field(default_factory=list)
    controls: List[Control] = field(default_factory=list)
    actors: List[Actor] = field(default_factory=list)
    standards: List[Standard] = field(default_factory=list)
    document_sections: List[DocumentSection] = field(default_factory=list)
    lifecycle_phases: List[LifecyclePhase] = field(default_factory=list)
    # Relationships
    hazard_control_links: List[HazardControlLink] = field(default_factory=list)
    hazard_actor_links: List[HazardActorLink] = field(default_factory=list)
    hazard_category_links: List[HazardCategoryLink] = field(default_factory=list)
    hazard_lifecycle_links: List[HazardLifecycleLink] = field(default_factory=list)
    control_standard_links: List[ControlStandardLink] = field(default_factory=list)
    control_document_links: List[ControlDocumentLink] = field(default_factory=list)


class ExcelParser:
    """Parser for Risk Analysis Excel files."""
    
    ACTOR_COLUMNS = ['Product', 'User', 'Third party', 'Service', 'Environment']
    
    # Known standards patterns
    STANDARD_PATTERNS = [
        (r'61010-1', 'IEC 61010-1', 'Safety requirements for electrical equipment'),
        (r'61326', 'IEC 61326', 'EMC requirements'),
        (r'61000-3-2', 'IEC 61000-3-2', 'Harmonic current emissions'),
        (r'61000-3-3', 'IEC 61000-3-3', 'Voltage fluctuations'),
        (r'2002/96/EC', 'WEEE Directive 2002/96/EC', 'Waste Electrical and Electronic Equipment'),
        (r'RoHS', 'RoHS', 'Restriction of Hazardous Substances'),
        (r'DOT', 'DOT', 'Department of Transportation'),
    ]
    
    # Lifecycle phase keywords
    LIFECYCLE_KEYWORDS = {
        'Operation': ['operation', 'operating', 'use', 'using', 'run', 'running'],
        'Maintenance': ['maintenance', 'service', 'repair', 'servicing'],
        'Installation': ['installation', 'install', 'setup', 'set-up'],
        'Transport': ['transport', 'shipping', 'moving'],
        'Storage': ['storage', 'storing', 'stored'],
        'Cleaning': ['cleaning', 'clean'],
        'Disposal': ['disposal', 'dispose', 'end of life', 'decommission'],
    }
    
    # Q_Source mappings to lifecycle
    Q_SOURCE_LIFECYCLE = {
        'A': 'Operation',      # Application/Use
        'D': 'Design',         # Design phase
        'SR': 'Maintenance',   # Service Review
        'PRD': 'Production',   # Production
        'W/S': 'Maintenance',  # Workshop/Service
        'V': 'Verification',   # Verification
        'PM': 'Production',    # Production/Manufacturing
        'PDM': 'Design',       # Product Design/Manufacturing
        'H': 'Operation',      # Hazard during operation
    }
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.xl = pd.ExcelFile(file_path)
        self.parsed_data = ParsedData()
        self._standards_found: Set[str] = set()
        self._documents_found: Set[str] = set()
        self._lifecycles_found: Set[str] = set()
        
    def parse(self) -> ParsedData:
        """Parse all sheets and return structured data."""
        # Find the sheet with most data (usually the most complete one)
        best_sheet = self._find_best_sheet()
        df = pd.read_excel(self.file_path, sheet_name=best_sheet)
        
        # Identify table boundaries
        table_ranges = self._identify_tables(df)
        
        # Parse each table
        if 'table1' in table_ranges:
            self._parse_basic_hazards(df, table_ranges['table1'])
        
        if 'table2' in table_ranges:
            self._parse_risk_analysis(df, table_ranges['table2'])
        
        if 'table3' in table_ranges:
            self._parse_prevention_measures(df, table_ranges['table3'])
        
        # Add standard actors
        for actor_name in self.ACTOR_COLUMNS:
            self.parsed_data.actors.append(Actor(name=actor_name))
        
        # Add collected standards
        for std_id in self._standards_found:
            for pattern, name, desc in self.STANDARD_PATTERNS:
                if std_id == name:
                    self.parsed_data.standards.append(Standard(id=std_id, name=desc))
                    break
        
        # Add collected document sections
        for doc_id in self._documents_found:
            doc_type = self._extract_document_type(doc_id)
            self.parsed_data.document_sections.append(
                DocumentSection(id=doc_id, name=doc_id, document_type=doc_type)
            )
        
        # Add collected lifecycle phases
        for phase in self._lifecycles_found:
            self.parsed_data.lifecycle_phases.append(LifecyclePhase(name=phase))
        
        return self.parsed_data
    
    def _find_best_sheet(self) -> str:
        """Find the sheet with the most rows (most complete data)."""
        max_rows = 0
        best_sheet = self.xl.sheet_names[0]
        
        for sheet in self.xl.sheet_names:
            df = pd.read_excel(self.file_path, sheet_name=sheet)
            if len(df) > max_rows:
                max_rows = len(df)
                best_sheet = sheet
        
        return best_sheet
    
    def _identify_tables(self, df: pd.DataFrame) -> Dict[str, Tuple[int, int]]:
        """Identify the start and end rows of each table."""
        tables = {}
        
        for i, row in df.iterrows():
            first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            
            if 'Table 1' in first_col or 'Basic Hazards' in first_col:
                tables['table1_start'] = i
            elif 'Table 2' in first_col or 'Risk Analysis' in first_col:
                if 'table1_start' in tables:
                    tables['table1'] = (tables['table1_start'], i - 1)
                tables['table2_start'] = i
            elif 'Table 3' in first_col or 'Prevention Measures' in first_col:
                if 'table2_start' in tables:
                    tables['table2'] = (tables['table2_start'], i - 1)
                tables['table3_start'] = i
        
        # Handle last table
        if 'table3_start' in tables:
            tables['table3'] = (tables['table3_start'], len(df) - 1)
        elif 'table2_start' in tables and 'table2' not in tables:
            tables['table2'] = (tables['table2_start'], len(df) - 1)
        
        return tables
    
    def _parse_basic_hazards(self, df: pd.DataFrame, row_range: Tuple[int, int]):
        """Parse Table 1: Basic Hazards to extract categories and actor associations."""
        start, end = row_range
        
        # Find header row
        header_row = None
        actor_col_indices = {}
        
        for i in range(start, min(start + 10, end)):
            row = df.iloc[i]
            first_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            
            if 'Hazard' in first_col:
                header_row = i
                # Find actor columns
                for j, col_val in enumerate(row):
                    col_str = str(col_val) if pd.notna(col_val) else ""
                    for actor in self.ACTOR_COLUMNS:
                        if actor.lower() in col_str.lower():
                            actor_col_indices[actor] = j
                break
        
        if header_row is None:
            return
        
        current_category = None
        
        for i in range(header_row + 1, end + 1):
            row = df.iloc[i]
            hazard_col = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            source_col = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
            
            if not hazard_col and not source_col:
                continue
            
            # Check if this is a new hazard category (numbered like "3 Electromagnetic...")
            category_match = re.match(r'^(\d+)\s+(.+)$', hazard_col.strip())
            if category_match:
                category_id = category_match.group(1)
                category_name = hazard_col.strip()
                
                # Create new category
                category = HazardCategory(id=category_id, name=category_name)
                exists = any(c.id == category_id for c in self.parsed_data.hazard_categories)
                if not exists:
                    self.parsed_data.hazard_categories.append(category)
                current_category = category_id
    
    def _parse_risk_analysis(self, df: pd.DataFrame, row_range: Tuple[int, int]):
        """Parse Table 2: Potential Hazard Assessment / Risk Analysis."""
        start, end = row_range
        
        # Find header row
        header_row = None
        for i in range(start, min(start + 10, end)):
            first_col = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
            if 'No.' in first_col or first_col.strip() == 'No':
                header_row = i
                break
        
        if header_row is None:
            return
        
        # Map column indices
        headers = df.iloc[header_row].tolist()
        col_map = self._map_columns(headers)
        
        cause_counter = 0
        consequence_counter = 0
        
        for i in range(header_row + 1, end + 1):
            row = df.iloc[i]
            
            # Get hazard ID
            hazard_id = str(row.iloc[col_map.get('no', 0)]) if pd.notna(row.iloc[col_map.get('no', 0)]) else ""
            hazard_id = hazard_id.strip()
            
            # Skip non-hazard rows
            if not hazard_id or not re.match(r'^\d+\.?\d*$', hazard_id):
                continue
            
            # Extract hazard data
            hazard_name = str(row.iloc[col_map.get('hazard', 1)]) if pd.notna(row.iloc[col_map.get('hazard', 1)]) else ""
            
            h_type = str(row.iloc[col_map.get('h_type', 4)]) if col_map.get('h_type') and pd.notna(row.iloc[col_map.get('h_type', 4)]) else ""
            q_source = str(row.iloc[col_map.get('q_source', 5)]) if col_map.get('q_source') and pd.notna(row.iloc[col_map.get('q_source', 5)]) else ""
            
            # Risk scores
            p_init = self._parse_int(row.iloc[col_map.get('p_init', 6)]) if col_map.get('p_init') else None
            s_init = self._parse_int(row.iloc[col_map.get('s_init', 7)]) if col_map.get('s_init') else None
            r_init = str(row.iloc[col_map.get('r_init', 8)]) if col_map.get('r_init') and pd.notna(row.iloc[col_map.get('r_init', 8)]) else ""
            
            p_final = self._parse_int(row.iloc[col_map.get('p_final', 10)]) if col_map.get('p_final') else None
            s_final = self._parse_int(row.iloc[col_map.get('s_final', 11)]) if col_map.get('s_final') else None
            r_final = str(row.iloc[col_map.get('r_final', 12)]) if col_map.get('r_final') and pd.notna(row.iloc[col_map.get('r_final', 12)]) else ""
            
            # Create hazard
            hazard = Hazard(
                id=hazard_id,
                name=hazard_name,
                h_type=h_type,
                q_source=q_source,
                p_init=p_init,
                s_init=s_init,
                r_init=r_init,
                p_final=p_final,
                s_final=s_final,
                r_final=r_final
            )
            self.parsed_data.hazards.append(hazard)
            
            # Extract cause
            cause_text = str(row.iloc[col_map.get('cause', 2)]) if col_map.get('cause') and pd.notna(row.iloc[col_map.get('cause', 2)]) else ""
            if cause_text and cause_text != 'nan':
                cause_counter += 1
                cause = Cause(id=f"C{cause_counter}", description=cause_text, hazard_id=hazard_id)
                self.parsed_data.causes.append(cause)
            
            # Extract consequence
            consequence_text = str(row.iloc[col_map.get('consequence', 3)]) if col_map.get('consequence') and pd.notna(row.iloc[col_map.get('consequence', 3)]) else ""
            if consequence_text and consequence_text != 'nan':
                consequence_counter += 1
                consequence = Consequence(id=f"CON{consequence_counter}", description=consequence_text, hazard_id=hazard_id)
                self.parsed_data.consequences.append(consequence)
            
            # Extract control links from prevention measures column
            prevention_text = str(row.iloc[col_map.get('prevention', 9)]) if col_map.get('prevention') and pd.notna(row.iloc[col_map.get('prevention', 9)]) else ""
            control_ids = self._extract_control_ids(prevention_text)
            
            # Calculate risk reduction for each control link
            p_reduction = (p_init - p_final) if p_init and p_final else None
            s_reduction = (s_init - s_final) if s_init and s_final else None
            
            for control_id in control_ids:
                link = HazardControlLink(
                    hazard_id=hazard_id, 
                    control_id=control_id,
                    p_reduction=p_reduction,
                    s_reduction=s_reduction
                )
                self.parsed_data.hazard_control_links.append(link)
            
            # Link hazard to category based on ID prefix
            category_id = hazard_id.split('.')[0]
            link = HazardCategoryLink(hazard_id=hazard_id, category_id=category_id)
            self.parsed_data.hazard_category_links.append(link)
            
            # Extract lifecycle phases from q_source
            lifecycle_phases = self._extract_lifecycle_phases(q_source, hazard_name, cause_text)
            for phase in lifecycle_phases:
                self._lifecycles_found.add(phase)
                link = HazardLifecycleLink(hazard_id=hazard_id, lifecycle_phase=phase)
                self.parsed_data.hazard_lifecycle_links.append(link)
            
            # Extract actors affected (from hazard name and consequence)
            actors = self._extract_actors(hazard_name, consequence_text)
            for actor in actors:
                link = HazardActorLink(hazard_id=hazard_id, actor_name=actor)
                self.parsed_data.hazard_actor_links.append(link)
    
    def _parse_prevention_measures(self, df: pd.DataFrame, row_range: Tuple[int, int]):
        """Parse Table 3: Prevention Measures (Controls)."""
        start, end = row_range
        
        # Find header row
        header_row = None
        for i in range(start, min(start + 10, end)):
            first_col = str(df.iloc[i, 0]) if pd.notna(df.iloc[i, 0]) else ""
            if 'M No' in first_col or first_col.strip() == 'M No.':
                header_row = i
                break
        
        if header_row is None:
            return
        
        for i in range(header_row + 1, end + 1):
            row = df.iloc[i]
            
            control_id = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            control_id = control_id.strip()
            
            # Skip if not a valid control ID (M1, M2, etc.)
            if not re.match(r'^M\d+$', control_id):
                continue
            
            description = str(row.iloc[1]) if pd.notna(row.iloc[1]) else ""
            implementation_ref = str(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else ""
            
            if description and description != 'nan':
                control = Control(
                    id=control_id,
                    description=description,
                    implementation_reference=implementation_ref if implementation_ref != 'nan' else ""
                )
                self.parsed_data.controls.append(control)
                
                # Extract standards from implementation reference
                standards = self._extract_standards(implementation_ref + " " + description)
                for std_id in standards:
                    self._standards_found.add(std_id)
                    link = ControlStandardLink(control_id=control_id, standard_id=std_id)
                    self.parsed_data.control_standard_links.append(link)
                
                # Extract document sections
                doc_sections = self._extract_document_sections(implementation_ref)
                for doc_id in doc_sections:
                    self._documents_found.add(doc_id)
                    link = ControlDocumentLink(control_id=control_id, document_id=doc_id)
                    self.parsed_data.control_document_links.append(link)
    
    def _map_columns(self, headers: List) -> Dict[str, int]:
        """Map column headers to indices."""
        col_map = {}
        
        for i, header in enumerate(headers):
            header_str = str(header).lower().strip() if pd.notna(header) else ""
            
            if 'no.' in header_str or header_str == 'no':
                col_map['no'] = i
            elif 'potential hazard' in header_str or header_str == 'potential hazard':
                col_map['hazard'] = i
            elif 'cause' in header_str:
                col_map['cause'] = i
            elif 'consequence' in header_str:
                col_map['consequence'] = i
            elif 'h_type' in header_str or header_str == 'h_type':
                col_map['h_type'] = i
            elif 'q_source' in header_str or header_str == 'q_source':
                col_map['q_source'] = i
            elif 'p_init' in header_str:
                col_map['p_init'] = i
            elif 's_init' in header_str:
                col_map['s_init'] = i
            elif 'r_init' in header_str:
                col_map['r_init'] = i
            elif 'prevention' in header_str:
                col_map['prevention'] = i
            elif 'p_final' in header_str:
                col_map['p_final'] = i
            elif 's_final' in header_str:
                col_map['s_final'] = i
            elif 'r_final' in header_str:
                col_map['r_final'] = i
        
        return col_map
    
    def _extract_control_ids(self, text: str) -> List[str]:
        """Extract control IDs (M1, M2, etc.) from prevention measures text."""
        if not text or text == 'nan':
            return []
        
        # Find all M followed by numbers
        matches = re.findall(r'M\d+', text)
        return list(set(matches))  # Remove duplicates
    
    def _extract_standards(self, text: str) -> List[str]:
        """Extract standard references from text."""
        if not text or text == 'nan':
            return []
        
        standards = []
        for pattern, name, desc in self.STANDARD_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                standards.append(name)
        
        return list(set(standards))
    
    def _extract_document_sections(self, text: str) -> List[str]:
        """Extract document section references from implementation reference."""
        if not text or text == 'nan':
            return []
        
        sections = []
        
        # Pattern for OI sections
        oi_match = re.findall(r'OI\s+(?:Section,?\s*)?([^,.]+)', text, re.IGNORECASE)
        for match in oi_match:
            sections.append(f"OI Section: {match.strip()}")
        
        # Pattern for PSS references
        pss_match = re.findall(r'PSS\s+[\d-]+', text, re.IGNORECASE)
        sections.extend(pss_match)
        
        # Pattern for QMS/QP references
        qms_match = re.findall(r'Q[MP]\s+[\d.]+', text, re.IGNORECASE)
        sections.extend(qms_match)
        
        # Pattern for UL reports
        ul_match = re.findall(r'UL\s+report\s+[^\s,]+', text, re.IGNORECASE)
        sections.extend(ul_match)
        
        return list(set(sections))
    
    def _extract_document_type(self, doc_id: str) -> str:
        """Extract document type from document ID."""
        if 'OI' in doc_id.upper():
            return 'Operating Instructions'
        elif 'PSS' in doc_id.upper():
            return 'Process Sequence Sheet'
        elif 'QM' in doc_id.upper() or 'QP' in doc_id.upper():
            return 'Quality Management'
        elif 'UL' in doc_id.upper():
            return 'UL Certification'
        return 'Other'
    
    def _extract_lifecycle_phases(self, q_source: str, hazard_name: str, cause_text: str) -> List[str]:
        """Extract lifecycle phases from q_source and context."""
        phases = set()
        
        # Extract from q_source codes
        if q_source and q_source != 'nan':
            # Split by comma or space
            codes = re.split(r'[,\s]+', q_source)
            for code in codes:
                code = code.strip().upper()
                if code in self.Q_SOURCE_LIFECYCLE:
                    phases.add(self.Q_SOURCE_LIFECYCLE[code])
        
        # Extract from hazard name and cause text
        combined_text = f"{hazard_name} {cause_text}".lower()
        for phase, keywords in self.LIFECYCLE_KEYWORDS.items():
            for keyword in keywords:
                if keyword in combined_text:
                    phases.add(phase)
                    break
        
        # Default to Operation if nothing found
        if not phases:
            phases.add('Operation')
        
        return list(phases)
    
    def _extract_actors(self, hazard_name: str, consequence_text: str) -> List[str]:
        """Extract affected actors from hazard and consequence text."""
        actors = set()
        combined_text = f"{hazard_name} {consequence_text}".lower()
        
        actor_keywords = {
            'User': ['user', 'operator', 'personnel', 'person', 'worker'],
            'Product': ['product', 'sample', 'material', 'device'],
            'Service': ['service', 'maintenance', 'technician', 'repair'],
            'Third party': ['third party', 'visitor', 'bystander'],
            'Environment': ['environment', 'surroundings', 'area', 'lab', 'laboratory'],
        }
        
        for actor, keywords in actor_keywords.items():
            for keyword in keywords:
                if keyword in combined_text:
                    actors.add(actor)
                    break
        
        # Default to User if nothing specific found
        if not actors:
            actors.add('User')
        
        return list(actors)
    
    def _parse_int(self, value) -> Optional[int]:
        """Safely parse integer from cell value."""
        if pd.isna(value):
            return None
        try:
            # Handle values like "5 Frequent" by extracting just the number
            val_str = str(value).strip()
            match = re.match(r'^(\d+)', val_str)
            if match:
                return int(match.group(1))
            return int(float(value))
        except (ValueError, TypeError):
            return None


def parse_excel(file_path: str) -> ParsedData:
    """Convenience function to parse an Excel file."""
    parser = ExcelParser(file_path)
    return parser.parse()


if __name__ == "__main__":
    # Test parsing
    import sys
    if len(sys.argv) > 1:
        data = parse_excel(sys.argv[1])
        print(f"Parsed {len(data.hazards)} hazards")
        print(f"Parsed {len(data.controls)} controls")
        print(f"Parsed {len(data.causes)} causes")
        print(f"Parsed {len(data.consequences)} consequences")
        print(f"Parsed {len(data.standards)} standards")
        print(f"Parsed {len(data.document_sections)} document sections")
        print(f"Parsed {len(data.lifecycle_phases)} lifecycle phases")
        print(f"Parsed {len(data.hazard_control_links)} hazard-control links")
        print(f"Parsed {len(data.hazard_actor_links)} hazard-actor links")
        print(f"Parsed {len(data.hazard_lifecycle_links)} hazard-lifecycle links")
        print(f"Parsed {len(data.control_standard_links)} control-standard links")
        print(f"Parsed {len(data.control_document_links)} control-document links")