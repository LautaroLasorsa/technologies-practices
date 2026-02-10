FIX44.xml Data Dictionary
=========================

The acceptor.cfg and initiator.cfg reference a FIX44.xml data dictionary file.
This file ships with the QuickFIX library and defines all FIX 4.4 message types,
fields, and validation rules.

When installed via vcpkg, the file is located at:

  %VCPKG_ROOT%/installed/x64-windows/share/quickfix/FIX44.xml

To use it, either:

  Option A: Copy it into this config/ directory.
    copy "%VCPKG_ROOT%\installed\x64-windows\share\quickfix\FIX44.xml" config\

  Option B: Update the DataDictionary path in acceptor.cfg and initiator.cfg
    to point to the vcpkg location.

  Option C: If the path above doesn't exist, check:
    %VCPKG_ROOT%/installed/x64-windows/share/quickfix/spec/FIX44.xml

The file is also available on GitHub:
  https://github.com/quickfix/quickfix/blob/master/spec/FIX44.xml
