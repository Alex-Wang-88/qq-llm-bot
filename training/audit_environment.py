"""Write machine and package evidence without environment values or credentials."""
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from common import LOCAL, REPO, configure, memory, save_json, stamp


def main():
    torch=configure()
    system=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',
        "[Console]::OutputEncoding=[Text.UTF8Encoding]::new(); $o=Get-CimInstance Win32_OperatingSystem; [ordered]@{os=$o|Select-Object Caption,Version,BuildNumber;cpu=Get-CimInstance Win32_Processor|Select-Object Name,NumberOfCores,NumberOfLogicalProcessors;gpu=Get-CimInstance Win32_VideoController|Select-Object Name,DriverVersion,DriverDate;ram_gib=$o.TotalVisibleMemorySize/1MB;free_ram_gib=$o.FreePhysicalMemory/1MB} | ConvertTo-Json -Depth 5"],text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW))
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    disk=shutil.disk_usage(REPO)
    report={'timestamp':stamp(),'system':system,'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
       'torch':torch.__version__,'hip':torch.version.hip,'devices':[str(torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())],
       'packages':packages,'disk_free_gib':disk.free/2**30,'memory':memory(torch),
       'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
       'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip()}
    save_json(LOCAL/'environment.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
