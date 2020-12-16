/**
 * Copyright (C) 2019 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#include "XclBinSignature.h"

#include <iostream>
#include <vector>

#ifndef _WIN32
  #include <openssl/cms.h>
  #include <openssl/pem.h>
#endif

#include "XclBinUtilities.h"
namespace XUtil = XclBinUtilities;

static bool
copyFile(const std::string & _src, const std::string _dest)
{
  XUtil::TRACE(XUtil::format("Copying file '%s' to '%s'", _src.c_str(), _dest.c_str()).c_str());

  std::ifstream src(_src.c_str(), std::ios::binary);
  std::ofstream dest(_dest.c_str(), std::ios::binary);
  dest << src.rdbuf();
  return src && dest;
}

static void
writeImageToFile(const char * _pBuffer, uint64_t _size, const std::string _sFile)
{
  XUtil::TRACE(XUtil::format("Writing 0x%lx bytes to the file: '%s'", _size, _sFile.c_str()).c_str());

  std::fstream oFile;
  oFile.open(_sFile, std::ifstream::out | std::ifstream::binary);
  if (!oFile.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for writing: " + _sFile;
    throw std::runtime_error(errMsg);
  }

  oFile.write(_pBuffer, _size);
  oFile.close();
}



void
getXclBinPKCSStats( const std::string& _xclBinFile,
                    XclBinPKCSImageStats& _xclBinPKCSImageStats) {
  // -- Initialize return values --
  _xclBinPKCSImageStats = { 0 };

  // Error checks
  if (_xclBinFile.empty()) {
    std::string errMsg = "ERROR: Missing xclbin file name to read from.";
    throw std::runtime_error(errMsg);
  }

  // -- Open the file for consumption --
  XUtil::TRACE("Reading xclbin binary file: " + _xclBinFile);
  std::fstream ifXclBin;
  ifXclBin.open(_xclBinFile, std::ifstream::in | std::ifstream::binary);
  if (!ifXclBin.is_open()) {
    std::string errMsg = "ERROR: Unable to open the file for reading: " + _xclBinFile;
    throw std::runtime_error(errMsg);
  }

  // Determine File Size
  ifXclBin.seekg(0, ifXclBin.end);
  _xclBinPKCSImageStats.file_size = ifXclBin.tellg();

  // Read in the header buffer
  axlf xclBinHeader;
  const unsigned int expectBufferSize = sizeof(axlf);

  ifXclBin.seekg(0);
  ifXclBin.read((char*)&xclBinHeader, sizeof(axlf));

  // -- Perform DRC checks

  // Error reading in the header
  if (ifXclBin.gcount() != expectBufferSize) {
    std::string errMsg = XUtil::format("ERROR: Occurred reading in the xclbin header.  Expected: 0x%lx, Actual: 0x%lx", expectBufferSize, ifXclBin.gcount());
    throw std::runtime_error(errMsg);
  }

  // -- Validate magic number
  std::string sMagicValue = XUtil::format("%s", xclBinHeader.m_magic).c_str();
  if (sMagicValue.compare("xclbin2") != 0) {
    std::string errMsg = XUtil::format("ERROR: The XCLBIN appears to be corrupted.  Expected magic value: 'xclbin2', actual: '%s'", sMagicValue.c_str());
    throw std::runtime_error(errMsg);
  }

  // We know it is an xclbin archive
  _xclBinPKCSImageStats.is_valid_xclbin_image = true;

  // Get signature information
  if (xclBinHeader.m_signature_length != -1) {
    _xclBinPKCSImageStats.is_PKCS_signed = true;
    _xclBinPKCSImageStats.signature_size = xclBinHeader.m_signature_length;
    _xclBinPKCSImageStats.signature_offset = xclBinHeader.m_header.m_length - xclBinHeader.m_signature_length;

    if (xclBinHeader.m_signature_length < -1) {
      throw std::runtime_error("ERROR: xclbin recorded signature length is corrupted.");
    }
  }

  // Get header file length
  _xclBinPKCSImageStats.image_size = xclBinHeader.m_header.m_length - xclBinHeader.m_signature_length;

  // Validate length
  uint64_t expectedFileSize = xclBinHeader.m_header.m_length;

  if (expectedFileSize != _xclBinPKCSImageStats.file_size) {
    std::string errMsg = XUtil::format("ERROR: Expected files size (0x%lx) does not match actual (0x%lx)", expectedFileSize, _xclBinPKCSImageStats.file_size);
    throw std::runtime_error(errMsg);
  }

  // We are done
  ifXclBin.close();
}


void signXclBinImage(const std::string& _fileOnDisk,
                     const std::string& _sPrivateKey,
                     const std::string& _sCertificate,
                     const std::string& _sDigestAlgorithm,
                     bool _bEnableDebugOutput)
// Equivalent openssl command:
//   openssl cms -md sha512 -nocerts -noattr -sign -signer certificate.cer -inkey private.key -binary -in u50.dts -outform der -out signature.openssl
#ifdef _WIN32
{
  throw std::runtime_error("ERROR: signXclBinImage not implemented on windows");
}
#else
{
  std::cout << "----------------------------------------------------------------------" << std::endl;
  std::cout << "Signing the archive file: '" + _fileOnDisk + "'" << std::endl;
  std::cout << "        Private key file: '" + _sPrivateKey + "'" << std::endl;
  std::cout << "        Certificate file: '" + _sCertificate + "'" << std::endl;
  std::cout << "        Digest Algorithm: '" + _sDigestAlgorithm + "'" << std::endl;


  XUtil::TRACE("SignXclBinImage");
  XUtil::TRACE("File On Disk: '" + _fileOnDisk + "'");
  XUtil::TRACE("Private Key: '" + _sPrivateKey + "'");
  XUtil::TRACE("Certificate: '" + _sCertificate + "'");

  // -- Do some DRC checks on the image
  // Is the image on disk
  XclBinPKCSImageStats xclBinPKCSStats = { 0 };
  getXclBinPKCSStats(_fileOnDisk, xclBinPKCSStats);

  if (xclBinPKCSStats.is_PKCS_signed == true) {
    throw std::runtime_error("ERROR: Xclbin image is already signed. File: '" + _fileOnDisk + "'");
  }

  // *** Calculate the signature **

  std::cout << "Calculating signature..." << std::endl;

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sDbgOriginalCopy = _fileOnDisk + ".sign_dbg.original";
    copyFile(_fileOnDisk, sDbgOriginalCopy);
  }

  // -- Have openssl point to the xclbin image on disk
  BIO* bmRead = BIO_new_file(_fileOnDisk.c_str(), "rb");

  if (bmRead == nullptr) {
    throw std::runtime_error("ERROR: File missing: '" + _fileOnDisk + "'");
  }

  // -- Read the private key --
  BIO* bmPrivateKey = BIO_new_file(_sPrivateKey.c_str(), "rb");
  if (bmPrivateKey == nullptr) {
    throw std::runtime_error("ERROR: File missing: '" + _sPrivateKey + "'");
  }

  EVP_PKEY* privateKey = PEM_read_bio_PrivateKey(bmPrivateKey, NULL, NULL, NULL);
  if (privateKey == nullptr) {
    throw std::runtime_error("ERROR: Cannot create private key object.");
  }

  BIO_free(bmPrivateKey);

  // -- Read the certificate --
  BIO* bmCertificate = BIO_new_file(_sCertificate.c_str(), "rb");
  if (bmCertificate == nullptr) {
    throw std::runtime_error("ERROR: File missing: '" + _sCertificate + "'");
  }

  X509* x509 = PEM_read_bio_X509(bmCertificate, NULL, NULL, NULL);
  if ((x509 == nullptr) && (BIO_seek(bmCertificate, 0) != -1)) {
    // Try reading in the certificate as DER file instead of PEM file. DER file is
    // default for DKMS generated UEFI secure boot certificates.
    x509 = d2i_X509_bio(bmCertificate, NULL);
  }

  if (x509 == nullptr) {
    throw std::runtime_error("ERROR: Cannot create certificate key object.");
  }

  BIO_free(bmCertificate);

  // -- Obtain the digest algorithm --
  OpenSSL_add_all_digests();
  const EVP_MD* digestAlgorithm = EVP_get_digestbyname(_sDigestAlgorithm.c_str());

  if (digestAlgorithm == nullptr) {
    std::string errMsg = XUtil::format("ERROR: Invalid digest algorithm: '%s'", _sDigestAlgorithm.c_str());
    throw std::runtime_error(errMsg);
  }

  // -- Prepare CMS content and signer info --
  CMS_ContentInfo* cmsContentInfo = CMS_sign(NULL, NULL, NULL, NULL,
                                             CMS_NOCERTS | CMS_PARTIAL | CMS_BINARY |
                                             CMS_DETACHED | CMS_STREAM);
  if (cmsContentInfo == nullptr) {
    throw std::runtime_error("ERROR: Could not obtain CMS content info");
  }

  CMS_SignerInfo* cmsSignerInfo = CMS_add1_signer(cmsContentInfo, x509, privateKey, digestAlgorithm,
                                                  CMS_NOCERTS | CMS_BINARY |
                                                  CMS_NOSMIMECAP | CMS_NOATTR);

  if (cmsSignerInfo == nullptr) {
    throw std::runtime_error("ERROR: Could not obtain CMS signer info");
  }

  // -- We are ready to tie it all together --
  if (CMS_final(cmsContentInfo, bmRead, NULL, CMS_NOCERTS | CMS_BINARY) < 0) {
    throw std::runtime_error("ERROR: In finalizing the CMS content.");
  }

  // We are done close the handles
  BIO_free(bmRead);

  // -- Get the signature --
  BIO* bmMem = BIO_new(BIO_s_mem());
  if (i2d_CMS_bio_stream(bmMem, cmsContentInfo, NULL, 0) < 0) {
    throw std::runtime_error("ERROR: Writing to the signature.bin to the in-memory buffer");
  }

  BUF_MEM *bufMem = nullptr;
  BIO_get_mem_ptr(bmMem, &bufMem);
  XUtil::TRACE_BUF("Signature", bufMem->data, bufMem->length);

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sSignatureFile = _fileOnDisk + ".sign_dbg.signature";
    XUtil::TRACE("Writing signature image");
    writeImageToFile(bufMem->data, bufMem->length, sSignatureFile);
  }

  // ** Now update the xclbin archive image **
  axlf xclBinHeader = {0};
  {
    std::fstream iofXclBin;
    iofXclBin.open(_fileOnDisk, std::ios::in | std::ios::out | std::ios::binary);
    if (!iofXclBin.is_open()) {
      std::string errMsg = "ERROR: Unable to open the file for reading / writing: " + _fileOnDisk;
      throw std::runtime_error(errMsg);
    }

    // -- Update the header --
    // Get the header
    iofXclBin.seekg(0);
    iofXclBin.read((char*)&xclBinHeader, sizeof(axlf));

    // Update the signature length
    xclBinHeader.m_signature_length = bufMem->length;
    XUtil::TRACE(XUtil::format("Setting the signature length to: 0x%x", xclBinHeader.m_signature_length).c_str());

    // Update header
    XUtil::TRACE(XUtil::format("Header length prior to signature: 0x%x", xclBinHeader.m_header.m_length ).c_str());
    xclBinHeader.m_header.m_length += (uint64_t) xclBinHeader.m_signature_length;
    XUtil::TRACE(XUtil::format("Header length with signature: 0x%x", xclBinHeader.m_header.m_length ).c_str());

    // All is good, write out the new header
    iofXclBin.seekg(0);
    iofXclBin.write((char*)&xclBinHeader, sizeof(axlf));

    iofXclBin.close();
  }

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sDbgOriginalCopy = _fileOnDisk + ".sign_dbg.modified_header";
    copyFile(_fileOnDisk, sDbgOriginalCopy);
  }

  // Now add the signature
  {
    std::fstream iofXclBin;
    iofXclBin.open(_fileOnDisk, std::ios::in | std::ios::out | std::ios::binary);
    if (!iofXclBin.is_open()) {
      std::string errMsg = "ERROR: Unable to open the file for reading / writing: " + _fileOnDisk;
      throw std::runtime_error(errMsg);
    }

    iofXclBin.seekg(0, iofXclBin.end);
    iofXclBin.write(bufMem->data, bufMem->length);

    // Check header size with actual size of file
    iofXclBin.seekg(0, iofXclBin.end);
    uint64_t fileSize = iofXclBin.tellg();

    if (fileSize != xclBinHeader.m_header.m_length) {
      std::string errMsg = XUtil::format("ERROR: xclbin file size (0x%lx) doesn't match expected header size length (0x%lx).", fileSize, xclBinHeader.m_header.m_length);
      throw std::runtime_error(errMsg);
    }

    // And we are done
    iofXclBin.close();
  }

  std::cout << "Signature calculated and added successfully to the file: '" << _fileOnDisk << "'" << std::endl;
  std::cout << "----------------------------------------------------------------------" << std::endl;
}
#endif

void
dumpSignatureFile(const std::string & _fileOnDisk,
                  const std::string & _signatureFile)
{
  XUtil::TRACE("Dump signature from xclbin archive");
  XUtil::TRACE("File On Disk: '" + _fileOnDisk + "'");
  XUtil::TRACE("Signature File: '" + _signatureFile + "'");

  // -- See if the image is signed
  XclBinPKCSImageStats xclBinPKCSStats = { 0 };
  getXclBinPKCSStats(_fileOnDisk, xclBinPKCSStats);

  if (xclBinPKCSStats.is_PKCS_signed == false) {
    throw std::runtime_error("ERROR: Xclbin image is not signed. File: '" + _fileOnDisk + "'");
  }

  XUtil::TRACE(XUtil::format("Signature offset: 0x%lx, length: 0x%lx", xclBinPKCSStats.signature_offset, xclBinPKCSStats.signature_size).c_str());

  //-- Read just the signature
  std::ifstream ifs(_fileOnDisk, std::ios::binary | std::ios::ate);

  // Reserve memory for the signature
  std::vector<char> memImage(xclBinPKCSStats.signature_size);

  // Go to the start of the signature
  ifs.seekg(xclBinPKCSStats.signature_offset, std::ios::beg);

  // Read in the signature
  ifs.read(memImage.data(), xclBinPKCSStats.signature_size);

  // Now write it out
  XUtil::TRACE("Writing signature file");
  writeImageToFile(memImage.data(), xclBinPKCSStats.signature_size, _signatureFile);
}


void verifyXclBinImage(const std::string& _fileOnDisk,
                       const std::string& _sCertificate,
                       bool _bEnableDebugOutput)

// Equivalent openssl command:
// openssl smime -verify -in signature.openssl.small -inform DER -content u50.dts -noverify -certfile certificate.cer -binary > /dev/null
#ifdef _WIN32
{
  throw std::runtime_error("ERROR: verifyXclBinImage not implemented on windows");
}
#else
{
  std::cout << "----------------------------------------------------------------------" << std::endl;
  std::cout << "Verifying signature for archive file: '" + _fileOnDisk + "'" << std::endl;
  std::cout << "                    Certificate file: '" + _sCertificate + "'" << std::endl;

  XUtil::TRACE("SignXclBinImage");
  XUtil::TRACE("File On Disk: '" + _fileOnDisk + "'");
  XUtil::TRACE("Certificate: '" + _sCertificate + "'");

  // -- Do some DRC checks on the image
  // Is the image on disk
  XclBinPKCSImageStats xclBinPKCSStats = { 0 };
  getXclBinPKCSStats(_fileOnDisk, xclBinPKCSStats);

  if (xclBinPKCSStats.is_PKCS_signed == false) {
    throw std::runtime_error("ERROR: Xclbin image is not signed. File: '" + _fileOnDisk + "'");
  }

  // ** Read in the memory image **
  std::cout << "Reading archive file..." << std::endl;

  std::ifstream ifs(_fileOnDisk, std::ios::binary | std::ios::ate);
  std::ifstream::pos_type pos = ifs.tellg();
  std::vector<char> memImage(pos);
  ifs.seekg(0, std::ios::beg);
  ifs.read(memImage.data(), pos);
  ifs.close();

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sDbgModifiedImage = _fileOnDisk + ".ver_dbg.modified_header";
    XUtil::TRACE("Writing verification modified header intermediate image");
    writeImageToFile(memImage.data(), xclBinPKCSStats.image_size, sDbgModifiedImage);
  }

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sDbgSignature = _fileOnDisk + ".ver_dbg.signature";
    XUtil::TRACE("Writing signature image");
    writeImageToFile(memImage.data() + xclBinPKCSStats.signature_offset, xclBinPKCSStats.signature_size, sDbgSignature);
  }

  // Update the header
  axlf *pXclBinHeader = (axlf *) memImage.data();

  // -- Change the header length to its original size when signed
  uint32_t signatureSize = pXclBinHeader->m_signature_length;
  XUtil::TRACE(XUtil::format("Signature length: 0x%x", pXclBinHeader->m_signature_length).c_str());

  XUtil::TRACE(XUtil::format("Header length prior to signature length removal: 0x%x", pXclBinHeader->m_header.m_length).c_str());
  pXclBinHeader->m_header.m_length -= pXclBinHeader->m_signature_length;
  XUtil::TRACE(XUtil::format("Header length prior after signature length removal: 0x%x", pXclBinHeader->m_header.m_length).c_str());

  // -- Change the signature length to -1 (since this was its signed value)
  pXclBinHeader->m_signature_length = -1;

  // -- Dump intermediate file
  if (_bEnableDebugOutput) {
    std::string sDbgModifiedImage = _fileOnDisk + ".ver_dbg.original";
    XUtil::TRACE("Writing original image used for signing");
    writeImageToFile(memImage.data(), xclBinPKCSStats.image_size, sDbgModifiedImage);
  }
  // ** Calculate the signature

  std::cout << "Validating signature..." << std::endl;

  BIO *bmImage = BIO_new_mem_buf(memImage.data(), pXclBinHeader->m_header.m_length);
  BIO *bmSignature = BIO_new_mem_buf((char *)(memImage.data() + pXclBinHeader->m_header.m_length), signatureSize);

  // -- Obtain the digest algorithm --
  OpenSSL_add_all_digests();

  // -- Read the certificate --
  BIO* bmCertificate = BIO_new_file(_sCertificate.c_str(), "rb");
  if (bmCertificate == nullptr) {
    throw std::runtime_error("ERROR: File missing: '" + _sCertificate + "'");
  }

  X509* x509 = PEM_read_bio_X509(bmCertificate, NULL, NULL, NULL);
  if ((x509 == nullptr) && (BIO_seek(bmCertificate, 0) != -1)) {
    // Try reading in the certificate as DER file instead of PEM file. DER file is
    // default for DKMS generated UEFI secure boot certificates.
    x509 = d2i_X509_bio(bmCertificate, NULL);
  }

  if (x509 == nullptr) {
    throw std::runtime_error("ERROR: Cannot create certificate key object.");
  }

  BIO_free(bmCertificate);

  // -- Set up trusted CA certificate store --
  X509_STORE* store = X509_STORE_new();

  if (!X509_STORE_add_cert(store, x509)) {
    throw std::runtime_error("ERROR: Can't add certificate.");
  }

  // -- Read in signature --
  PKCS7* p7 = d2i_PKCS7_bio(bmSignature, NULL);
  if (p7 == NULL) {
    std::string errMsg = XUtil::format("ERROR: Signature at offset 0x%lx is not valid.", pXclBinHeader->m_header.m_length);
    throw std::runtime_error(errMsg);
  }

  STACK_OF(X509) * ca_stack = sk_X509_new_null();
  sk_X509_push(ca_stack, x509);

  if (!PKCS7_verify(p7, ca_stack, store, bmImage, NULL, PKCS7_DETACHED |  PKCS7_BINARY | PKCS7_NOINTERN)) {
    std::cout << "Signed xclbin archive verification [FAILED]" << std::endl;
  } else {
    std::cout << "Signed xclbin archive verification [SUCCESSFUL]" << std::endl;
  }
  std::cout << "----------------------------------------------------------------------" << std::endl;
}
#endif
