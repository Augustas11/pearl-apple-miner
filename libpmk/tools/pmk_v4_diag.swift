// SPDX-License-Identifier: Apache-2.0
// Derives from Pearl fp8-scheme (ISC), pinned at f696760b259500ecb608469ea3953aeabbe78948.

import Foundation
import CPMK

func die(_ message: String) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(1)
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    die("usage: pmk_v4_diag metadata|fp8-selftest|write-admission <path> <exact_cells>")
}

var ctx: pmk_v4_context?
var error = [CChar](repeating: 0, count: 4096)
guard pmk_v4_init_diagnostic(&ctx, &error, UInt64(error.count)) == PMK_SUCCESS, let ctx else {
    die(String(cString: error))
}
defer { pmk_v4_destroy(ctx) }

switch args[1] {
case "metadata":
    var json = [CChar](repeating: 0, count: 8192)
    guard pmk_v4_admission_metadata(ctx, &json, UInt64(json.count)) == PMK_SUCCESS else {
        die("metadata failed")
    }
    print(String(cString: json))
case "fp8-selftest":
    let rc = pmk_v4_fp8_selftest(ctx, &error, UInt64(error.count))
    guard rc == PMK_SUCCESS else { die(String(cString: error)) }
    print("fp8-selftest passed")
case "write-admission":
    guard args.count == 4, let cells = UInt64(args[3]) else {
        die("usage: pmk_v4_diag write-admission <path> <exact_cells>")
    }
    let rc = args[2].withCString {
        pmk_v4_write_admission_record(ctx, $0, cells, &error, UInt64(error.count))
    }
    guard rc == PMK_SUCCESS else { die(String(cString: error)) }
    print("wrote \(args[2])")
default:
    die("unknown command \(args[1])")
}
