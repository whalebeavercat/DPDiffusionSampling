import os
import sys
from datetime import datetime
import logging
import colorama
from colorama import Fore, Style, Back
import textwrap
from tqdm.auto import tqdm

colorama.init(autoreset=True)

class LogLevel:
    """Log level constants"""
    SILENT = 0      # No output except critical errors
    ERROR = 1       # Only errors
    WARNING = 2     # Errors and warnings
    INFO = 3        # Normal verbosity (default)
    DEBUG = 4       # Detailed debug information
    TRACE = 5       # Very detailed tracing information

class DPLogger:
    """
    A logger class for DP-Denoiser that handles console output and file logging
    with different verbosity levels and color-coded formatting.
    """
    
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(self, name="dp-denoiser", log_dir="./logs", log_to_file=True, 
                 log_level=logging.INFO, console_level=logging.INFO, 
                 colored_output=True, line_width=100, verbosity=LogLevel.INFO):
        """
        Initialize the logger.
        
        Args:
            name: Name of the logger
            log_dir: Directory to save log files
            log_to_file: Whether to save logs to file
            log_level: Overall logging level for file logging
            console_level: Console output verbosity level
            colored_output: Whether to use colored output in console
            line_width: Maximum width for wrapped log messages
            verbosity: Custom verbosity level (0-5) controlling what gets shown
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        self.colored_output = colored_output
        self.line_width = line_width
        self.indent = "  "
        self.verbosity = verbosity
        self.progress_bars = {}
        
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_formatter = ColoredFormatter(colored=colored_output)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(log_dir, f"{name}-{timestamp}.log")
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter('[%(asctime)s] %(levelname)-8s: %(message)s', 
                                             datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            if self.verbosity >= LogLevel.INFO:
                self.info(f"Log file created at: {log_file}")
    
    def set_verbosity(self, level):
        """Set the verbosity level"""
        self.verbosity = level
    
    def get_verbosity(self):
        """Get the current verbosity level"""
        return self.verbosity
            
    def _format_message(self, message, indent_level=0):
        """Format message with proper indentation and line wrapping"""
        indent = self.indent * indent_level
        if isinstance(message, str):
            wrapped = textwrap.fill(message, width=self.line_width - len(indent),
                                   subsequent_indent=indent + "  ")
            return f"{indent}{wrapped}"
        return f"{indent}{message}"
    
    def debug(self, message, indent_level=0):
        """Log debug message if verbosity allows"""
        if self.verbosity >= LogLevel.DEBUG:
            self.logger.debug(self._format_message(message, indent_level))
    
    def info(self, message, indent_level=0):
        """Log info message if verbosity allows"""
        if self.verbosity >= LogLevel.INFO:
            self.logger.info(self._format_message(message, indent_level))
    
    def warning(self, message, indent_level=0):
        """Log warning message if verbosity allows"""
        if self.verbosity >= LogLevel.WARNING:
            self.logger.warning(self._format_message(message, indent_level))
    
    def error(self, message, indent_level=0):
        """Log error message if verbosity allows"""
        if self.verbosity >= LogLevel.ERROR:
            self.logger.error(self._format_message(message, indent_level))
    
    def critical(self, message, indent_level=0):
        """Log critical message (always shown)"""
        self.logger.critical(self._format_message(message, indent_level))
    
    def section(self, title):
        """Create a section header if verbosity allows"""
        if self.verbosity >= LogLevel.INFO:
            if self.colored_output:
                header = f"\n{Fore.BLUE}{Back.WHITE}{Style.BRIGHT} {title} {Style.RESET_ALL}"
                self.info(f"{header}\n{Fore.BLUE}{'=' * len(title) + '=='}{Style.RESET_ALL}")
            else:
                self.info(f"\n{title}\n{'=' * len(title) + '=='}")
    
    def subsection(self, title):
        """Create a subsection header if verbosity allows"""
        if self.verbosity >= LogLevel.INFO:
            if self.colored_output:
                self.info(f"\n{Fore.CYAN}{Style.BRIGHT}{title}{Style.RESET_ALL}")
                self.info(f"{Fore.CYAN}{'-' * len(title)}{Style.RESET_ALL}")
            else:
                self.info(f"\n{title}\n{'-' * len(title)}")
    
    def result(self, message, indent_level=0):
        """Log a result message (highlighted) - always shown at WARNING+ verbosity"""
        if self.verbosity >= LogLevel.WARNING:
            if self.colored_output:
                msg = f"{Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}"
                self.info(msg, indent_level)
            else:
                self.info(f"RESULT: {message}", indent_level)
    
    def parameter(self, name, value, indent_level=0):
        """Log a parameter name-value pair if verbosity allows"""
        if self.verbosity >= LogLevel.INFO:
            if self.colored_output:
                msg = f"{Fore.YELLOW}{name}{Style.RESET_ALL}: {value}"
                self.info(msg, indent_level)
            else:
                self.info(f"{name}: {value}", indent_level)
    
    def create_progress_bar(self, name, total, desc="Processing", unit="it"):
        """Create a progress bar for iterative operations"""
        if name in self.progress_bars:
            self.progress_bars[name].close()
        
        disable = self.verbosity >= LogLevel.DEBUG
        
        self.progress_bars[name] = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            disable=disable,
            leave=True
        )
        return self.progress_bars[name]
    
    def update_progress(self, name, n=1):
        """Update a progress bar by n steps"""
        if name in self.progress_bars:
            self.progress_bars[name].update(n)
            
    def close_progress(self, name):
        """Close a progress bar"""
        if name in self.progress_bars:
            self.progress_bars[name].close()
            del self.progress_bars[name]
    
    def log_beta_analysis_start(self, config, n_samples, rho=7):
        """Log the start of beta analysis with parameters"""
        if self.verbosity >= LogLevel.INFO:
            self.section(f"Beta Analysis for {config.dataset.upper()}")
            self.parameter("Dataset", config.dataset)
            self.parameter("Samples", n_samples, indent_level=1)
            self.parameter("Batch size", config.batch_size, indent_level=1)
            self.parameter("Timesteps", config.num_timesteps, indent_level=1)
            self.subsection("Hyperparameters")
            self.parameter("rho", rho, indent_level=1)
    
    def log_conditions(self, condition_I, condition_II):
        """
        Log beta condition bounds if verbosity allows
        
        Args:
            condition_I: Upper bound for beta (condition I)
            condition_II: Lower bound for beta (condition II)
            pwr: Power factor (1+2ρ)/(2ρ)
        """
        if self.verbosity >= LogLevel.INFO:
            self.subsection("Beta Conditions")
            self.parameter("Condition I (upper bound)", f"β ≤ {condition_I.item():.6e}", indent_level=1)
            self.parameter("Condition II (lower bound)", f"β > {condition_II.item():.6e}", indent_level=1)
            
    def log_beta_test_range(self, count, min_beta, max_beta):
        """Log the range of beta values being tested if verbosity allows"""
        if self.verbosity >= LogLevel.INFO:
            self.subsection("Beta Test Range")
            self.info(f"Testing {count} beta values from {min_beta:.6e} to {max_beta:.6e}")
    
    def log_beta_result(self, beta, epsilon, lambda_val):
        """Log the result for a specific beta value if verbosity allows"""
        if self.verbosity >= LogLevel.DEBUG:
            self.info(f"β = {beta:.6f}: ε = {epsilon:.6f}, λ = {lambda_val:.6f}")
    
    def log_beta_error(self, beta, error_msg):
        """Log error for a specific beta value if verbosity allows"""
        if self.verbosity >= LogLevel.WARNING:
            if self.colored_output:
                self.warning(f"Error with β = {beta:.6f}: {Fore.RED}{error_msg}{Style.RESET_ALL}")
            else:
                self.warning(f"Error with β = {beta:.6f}: {error_msg}")
    
    def log_optimal_lambda(self, lambda_val, epsilon, lambda_max):
        """Log optimal lambda result if verbosity allows"""
        if self.verbosity >= LogLevel.TRACE:
            self.info(f"λ = {lambda_val:.4f}: ε = {epsilon:.4f}: λ constraint = {lambda_max:.4f}")
    
    def log_optimal_beta(self, dataset_name, min_valid_eps, min_valid_beta):
        """Log optimal beta result - always shown at WARNING+ verbosity"""
        if self.verbosity >= LogLevel.WARNING:
            self.subsection("Optimal Result")
            self.result(f"Optimal β for {dataset_name}: {min_valid_beta:.6e}")
            self.result(f"Valid minimum ε: {min_valid_eps:.4f}")
    
    def log_no_valid_beta(self, dataset_name):
        """Log when no valid beta values are found - always shown at WARNING+ verbosity"""
        if self.verbosity >= LogLevel.WARNING:
            self.warning(f"No valid β values found for {dataset_name} that satisfy both bounds.")
    
    def log_all_nan_warning(self):
        """Log warning when all epsilon values are NaN - always shown at WARNING+ verbosity"""
        if self.verbosity >= LogLevel.WARNING:
            self.warning("All epsilon values are NaN. Cannot determine optimal beta.")
            self.warning("Consider relaxing the lambda constraints or adjusting parameters.", indent_level=1)


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    
    def __init__(self, colored=True):
        super().__init__()
        self.colored = colored
        
        if colored:
            self.level_colors = {
                logging.DEBUG: Fore.BLUE,
                logging.INFO: Fore.WHITE,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT
            }
        
    def format(self, record):
        if not self.colored:
            return record.getMessage()
            
        level_color = self.level_colors.get(record.levelno, Fore.WHITE)
        
        if record.levelno >= logging.WARNING:
            prefix = f"{level_color}{record.levelname}{Style.RESET_ALL}: "
            return f"{prefix}{record.getMessage()}"
        else:
            return f"{level_color}{record.getMessage()}{Style.RESET_ALL}"
